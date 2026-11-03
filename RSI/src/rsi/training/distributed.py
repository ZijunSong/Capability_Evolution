"""Multi-node / multi-GPU data-parallel plumbing for HarnessOPD training.

torchrun-launched ranks each collect their shard of the on-policy batch
serially on their own GPU, then synchronize:

1. global normalizers (Σ action tokens / Σ OPD weights) via one all_reduce,
2. batch-centered advantages via the global mean reward,
3. gradients via ``all_reduce(SUM)`` on trainable (LoRA) parameters,
4. one identical optimizer step per rank per batch.

Manual SUM grad reduction (not DDP mean-reduction) keeps the loss scaling
convention simple: local losses are already divided by the GLOBAL normalizers,
so summing gradients yields the exact full-batch gradient
(``test_distributed_normalization``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_from_env() -> DistInfo:
    """Initialize from torchrun env (RANK/WORLD_SIZE/LOCAL_RANK).

    Single-process fallback: world_size=1, device cuda if available else cpu.
    """
    import os

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        import torch
        import torch.distributed as dist

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        device = f"cuda:{local_rank}" if use_cuda else "cpu"
        return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank, device=device)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    return DistInfo(rank=0, world_size=1, local_rank=0, device=device)


def shard_items(items: Sequence[Any], rank: int, world_size: int) -> list[Any]:
    """Round-robin shard — deterministic across ranks (same input list)."""
    return [items[i] for i in range(rank, len(items), world_size)]


def all_reduce_sum(values: Sequence[float], dist_info: DistInfo) -> list[float]:
    if not dist_info.is_distributed:
        return [float(v) for v in values]
    import torch
    import torch.distributed as dist

    t = torch.tensor(list(values), dtype=torch.float64, device=dist_info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


def all_reduce_grads(model: Any, dist_info: DistInfo) -> None:
    """SUM-reduce gradients of trainable params across ranks (in place).

    Every trainable param gets a grad tensor on every rank first (zero-filled
    when that rank had no rows this batch) so the collective is uniform — a
    skipped all_reduce on any rank deadlocks the whole job.
    """
    if not dist_info.is_distributed:
        return
    import torch
    import torch.distributed as dist

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def barrier(dist_info: DistInfo) -> None:
    if not dist_info.is_distributed:
        return
    import torch.distributed as dist

    dist.barrier()
