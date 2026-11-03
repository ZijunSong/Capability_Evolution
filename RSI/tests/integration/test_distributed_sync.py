"""Distributed sync semantics under a real 2-process gloo group.

Covers the code the 32-GPU job exercises but single-process tests cannot:
uniform collectives (a rank with zero rows must not skip an all_reduce — that
deadlocks the job), SUM-grad correctness, and global normalizer math.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dist_worker(rank: int, world_size: int, port: int, out_q) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from rsi.training.distributed import all_reduce_grads, all_reduce_sum, init_from_env, shard_items

    dist_info = init_from_env()
    assert dist_info.world_size == world_size

    # 1) float all_reduce sums across ranks
    got = all_reduce_sum([float(rank + 1), 10.0], dist_info)

    # 2) shard coverage
    items = list(range(world_size * 3))
    mine = shard_items(items, rank, world_size)

    # 3) uniform grad collective: rank1 has NO grads (simulating a zero-row
    #    batch); rank0 has grads.  Without zero-fill this deadlocks.
    model = torch.nn.Linear(4, 4)
    if rank == 0:
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
    all_reduce_grads(model, dist_info)

    out_q.put(
        {
            "rank": rank,
            "sums": got,
            "shard": mine,
            "grad_sum": float(next(model.parameters()).grad.abs().sum()),
        }
    )


def test_two_rank_gloo_sync():
    import multiprocessing as mp

    world_size = 2
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_dist_worker, args=(r, world_size, port, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(world_size):
        item = q.get(timeout=120)
        results[item["rank"]] = item
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"rank exited with {p.exitcode}"

    # all_reduce_sum: [1+2, 10+10]
    for r in results.values():
        assert r["sums"] == [3.0, 20.0]
    # shard: disjoint cover
    seen = sorted(x for r in results.values() for x in r["shard"])
    assert seen == list(range(world_size * 3))
    # grad sync: rank1 (no local grads) received rank0's grads, not zeros-only
    assert results[0]["grad_sum"] > 0
    assert results[1]["grad_sum"] == results[0]["grad_sum"]


def test_zero_row_batch_no_deadlock_single_process():
    """world_size=1 zero-fill is a no-op and empty batches step safely."""
    from rsi.training.backend import GlobalNormalizers
    from rsi.training.distributed import DistInfo, all_reduce_grads
    from rsi.training.hf_reference import HFReferenceBackend
    from rsi.training.joint import MethodSpec, joint_update

    model = torch.nn.Linear(4, 4)
    backend = HFReferenceBackend(model=model, lr=1e-2)
    info = DistInfo(rank=0, world_size=1, local_rank=0, device="cpu")
    backend.grad_sync_hook = lambda: all_reduce_grads(model, info)
    report = joint_update(
        backend,
        rl_rows=[],
        opd_rows=[],
        spec=MethodSpec.resolve("harness_opd_rl"),
        normalizers=GlobalNormalizers(global_action_tokens=1, global_opd_weight=0.0),
    )
    assert report.n_optimizer_steps == 1  # empty batch still takes a safe step
