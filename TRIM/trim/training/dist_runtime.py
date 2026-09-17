"""Multi-node TRIM RL helpers.

Single-node ``run_train.py`` stays a one-process Scheme A loop (vLLM then HF
on the visible GPUs of that machine). Multi-node training launches one
Scheme A process per node (default ``--nproc-per-node 1``), shards the
on-policy query groups across ranks, gathers trajectories onto rank 0, and
lets rank 0 run the existing HF optimizer step. The next rollout reloads the
adapter from the shared ``--out`` directory.

``--out`` must be on a filesystem that every node can read and write.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence, TypeVar

T = TypeVar("T")

DIST_BACKEND_ENV = "TRIM_DIST_BACKEND"
DEFAULT_MASTER_PORT = 29500
DEFAULT_DIST_BACKEND = "gloo"
DEFAULT_RDZV_ID = "trim-rl"


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    node_rank: int = 0
    initialized: bool = False

    @property
    def is_coordinator(self) -> bool:
        return int(self.rank) == 0

    @property
    def distributed(self) -> bool:
        return int(self.world_size) > 1


@dataclass
class DistLaunchConfig:
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: int = DEFAULT_MASTER_PORT
    rdzv_backend: str = "c10d"
    rdzv_id: str = DEFAULT_RDZV_ID
    dist_backend: str = DEFAULT_DIST_BACKEND
    hosts: list[str] = field(default_factory=list)
    ssh_launch: bool = False
    rollout_replicas: int = 1
    already_worker: bool = False


_INFO: DistInfo | None = None


def reset_dist_info_for_tests() -> None:
    global _INFO
    _INFO = None


def under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def dist_info() -> DistInfo:
    if _INFO is not None:
        return _INFO
    if not under_torchrun():
        return DistInfo()
    return DistInfo(
        rank=int(os.environ.get("RANK") or 0),
        world_size=max(1, int(os.environ.get("WORLD_SIZE") or 1)),
        local_rank=int(os.environ.get("LOCAL_RANK") or 0),
        local_world_size=max(1, int(os.environ.get("LOCAL_WORLD_SIZE") or 1)),
        node_rank=int(os.environ.get("GROUP_RANK") or os.environ.get("NODE_RANK") or 0),
        initialized=False,
    )


def is_coordinator() -> bool:
    return dist_info().is_coordinator


def visible_cuda_count() -> int:
    """Count GPUs from CUDA_VISIBLE_DEVICES without initializing CUDA."""
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not visible or visible in {"-1", "none", "None"}:
        return 0
    return len([x.strip() for x in visible.split(",") if x.strip()])


def training_backend_from_argv(argv: Sequence[str]) -> str:
    for i, tok in enumerate(argv):
        if tok in {"--training-backend", "--training_backend"} and i + 1 < len(argv):
            return str(argv[i + 1]).lower().replace("-", "_")
        if tok.startswith("--training-backend="):
            return str(tok.split("=", 1)[1]).lower().replace("-", "_")
        if tok.startswith("--training_backend="):
            return str(tok.split("=", 1)[1]).lower().replace("-", "_")
    return "hf_debug"


def pin_local_cuda_device() -> dict[str, Any]:
    """Give each local rank a single visible GPU before torch/vLLM start.

    torchrun does not rewrite ``CUDA_VISIBLE_DEVICES``. With
    ``--nproc-per-node 8`` every rank would otherwise inherit the full node
    GPU list and vLLM would try to grab every card.
    """
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE") or 1)
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    payload: dict[str, Any] = {
        "pinned": False,
        "local_rank": local_rank,
        "local_world_size": local_world,
        "cuda_visible_devices": visible,
    }
    if local_world <= 1:
        return payload
    if visible and visible not in {"-1", "none", "None"}:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if local_rank >= len(ids):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but CUDA_VISIBLE_DEVICES has {len(ids)} ids: {visible!r}"
            )
        if len(ids) == 1:
            payload["cuda_visible_devices"] = ids[0]
            return payload
        chosen = ids[local_rank]
    else:
        chosen = str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    payload["pinned"] = True
    payload["cuda_visible_devices"] = chosen
    return payload


def init_dist_if_needed(*, backend: str | None = None, timeout_hours: float = 6.0) -> DistInfo:
    global _INFO
    if _INFO is not None and _INFO.initialized:
        return _INFO
    if not under_torchrun():
        _INFO = DistInfo()
        return _INFO
    import torch.distributed as dist

    resolved = str(
        backend
        or os.environ.get(DIST_BACKEND_ENV)
        or DEFAULT_DIST_BACKEND
    ).strip().lower()
    if not dist.is_initialized():
        dist.init_process_group(
            resolved,
            timeout=timedelta(hours=max(0.1, float(timeout_hours))),
        )
    _INFO = DistInfo(
        rank=int(dist.get_rank()),
        world_size=int(dist.get_world_size()),
        local_rank=int(os.environ.get("LOCAL_RANK") or 0),
        local_world_size=max(1, int(os.environ.get("LOCAL_WORLD_SIZE") or 1)),
        node_rank=int(os.environ.get("GROUP_RANK") or os.environ.get("NODE_RANK") or 0),
        initialized=True,
    )
    return _INFO


def barrier() -> None:
    info = dist_info()
    if info.world_size <= 1:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, *, src: int = 0) -> Any:
    info = dist_info()
    if info.world_size <= 1:
        return obj
    import torch.distributed as dist

    payload = [obj if int(info.rank) == int(src) else None]
    dist.broadcast_object_list(payload, src=int(src))
    return payload[0]


def shard_for_rank(items: Sequence[T], *, rank: int, world_size: int) -> list[T]:
    world = max(1, int(world_size))
    rnk = int(rank)
    if world <= 1:
        return list(items)
    if rnk < 0 or rnk >= world:
        raise ValueError(f"rank={rnk} is outside world_size={world}")
    return [item for i, item in enumerate(items) if i % world == rnk]


def interleave_round_robin(shards: Sequence[Sequence[T]]) -> list[T]:
    """Inverse of ``shard_for_rank`` / round-robin sharding."""
    parts = [list(s) for s in shards]
    merged: list[T] = []
    max_len = max((len(s) for s in parts), default=0)
    for i in range(max_len):
        for shard in parts:
            if i < len(shard):
                merged.append(shard[i])
    return merged


def gather_sharded_objects(
    local: Sequence[T],
    *,
    shard_dir: Path,
    tag: str,
) -> list[T]:
    """Write per-rank pickles to shared disk, rank 0 interleaves them.

    Trajectories can be large; file gather avoids NCCL object-size limits.
    """
    info = dist_info()
    items = list(local)
    if info.world_size <= 1:
        return items
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(tag))
    path = shard_dir / f"{safe_tag}.rank{int(info.rank):04d}.pkl"
    with path.open("wb") as handle:
        pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)
    barrier()
    if not info.is_coordinator:
        barrier()
        return []
    shards: list[list[T]] = []
    try:
        for rank in range(int(info.world_size)):
            shard_path = shard_dir / f"{safe_tag}.rank{rank:04d}.pkl"
            if not shard_path.is_file():
                raise FileNotFoundError(f"missing rollout shard {shard_path}")
            with shard_path.open("rb") as handle:
                shards.append(list(pickle.load(handle)))
    finally:
        for rank in range(int(info.world_size)):
            shard_path = shard_dir / f"{safe_tag}.rank{rank:04d}.pkl"
            try:
                shard_path.unlink(missing_ok=True)
            except OSError:
                pass
    merged = interleave_round_robin(shards)
    barrier()
    return merged


def all_gather_via_disk(local: Sequence[T], *, shard_dir: Path, tag: str) -> list[T]:
    """Like gather_sharded_objects, then share the merged list with every rank."""
    info = dist_info()
    merged = gather_sharded_objects(local, shard_dir=shard_dir, tag=tag)
    if info.world_size <= 1:
        return merged
    blob_path = Path(shard_dir) / f"{''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(tag))}.all.pkl"
    if info.is_coordinator:
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        with blob_path.open("wb") as handle:
            pickle.dump(list(merged), handle, protocol=pickle.HIGHEST_PROTOCOL)
    barrier()
    if not info.is_coordinator:
        with blob_path.open("rb") as handle:
            merged = list(pickle.load(handle))
    barrier()
    if info.is_coordinator:
        try:
            blob_path.unlink(missing_ok=True)
        except OSError:
            pass
    barrier()
    return merged


def parse_host_list(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]


def _env_int(*names: str) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        return int(raw)
    return None


def parse_dist_argv(argv: Sequence[str] | None = None) -> tuple[DistLaunchConfig, list[str]]:
    """Split multi-node launcher flags from the regular ``run_train.py`` argv."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--nnodes", type=int, default=None)
    parser.add_argument("--nproc-per-node", "--nproc_per_node", dest="nproc_per_node", type=int, default=None)
    parser.add_argument("--node-rank", "--node_rank", dest="node_rank", type=int, default=None)
    parser.add_argument("--master-addr", "--master_addr", dest="master_addr", default=None)
    parser.add_argument("--master-port", "--master_port", dest="master_port", type=int, default=None)
    parser.add_argument("--rdzv-backend", dest="rdzv_backend", default="c10d")
    parser.add_argument("--rdzv-id", dest="rdzv_id", default=DEFAULT_RDZV_ID)
    parser.add_argument(
        "--dist-backend",
        default=DEFAULT_DIST_BACKEND,
        help="torch.distributed backend for barriers/broadcast. gloo is the default (CPU objects).",
    )
    parser.add_argument(
        "--hosts",
        default="",
        help="Comma-separated hostnames. Used with --ssh-launch or to print per-node commands.",
    )
    parser.add_argument(
        "--ssh-launch",
        action="store_true",
        help="SSH from this process to every --hosts entry and start torchrun. Requires passwordless SSH.",
    )
    parser.add_argument(
        "--rollout-replicas",
        type=int,
        default=1,
        help=(
            "Intra-node vLLM replica count. 1 (default) = one TP engine on all local GPUs. "
            "For Qwen-4B prefer 8 with --tensor-parallel-size 1 so each card is an independent actor."
        ),
    )
    parser.add_argument(
        "--already-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    cfg_ns, rest = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    nnodes = cfg_ns.nnodes
    if nnodes is None:
        nnodes = _env_int("SLURM_NNODES", "PET_NNODES") or 1
    nproc = cfg_ns.nproc_per_node
    if nproc is None:
        nproc = _env_int("SLURM_NTASKS_PER_NODE", "NPROC_PER_NODE") or 1
    node_rank = cfg_ns.node_rank
    if node_rank is None:
        node_rank = _env_int("SLURM_NODEID", "GROUP_RANK", "NODE_RANK")
    master_addr = cfg_ns.master_addr or os.environ.get("MASTER_ADDR") or os.environ.get("SLURM_LAUNCH_NODE_IPADDR") or "127.0.0.1"
    master_port = cfg_ns.master_port or _env_int("MASTER_PORT") or DEFAULT_MASTER_PORT
    cfg = DistLaunchConfig(
        nnodes=max(1, int(nnodes)),
        nproc_per_node=max(1, int(nproc)),
        node_rank=None if node_rank is None else int(node_rank),
        master_addr=str(master_addr),
        master_port=int(master_port),
        rdzv_backend=str(cfg_ns.rdzv_backend or "c10d"),
        rdzv_id=str(cfg_ns.rdzv_id or DEFAULT_RDZV_ID),
        dist_backend=str(cfg_ns.dist_backend or DEFAULT_DIST_BACKEND),
        hosts=parse_host_list(cfg_ns.hosts),
        ssh_launch=bool(cfg_ns.ssh_launch),
        rollout_replicas=max(1, int(cfg_ns.rollout_replicas or 1)),
        already_worker=bool(cfg_ns.already_worker) or under_torchrun(),
    )
    return cfg, list(rest)


def torchrun_cmd(
    *,
    script: str | Path,
    train_argv: Sequence[str],
    cfg: DistLaunchConfig,
    node_rank: int,
    python_exe: str | None = None,
) -> list[str]:
    exe = python_exe or sys.executable
    nnodes = max(1, int(cfg.nnodes))
    cmd = [
        exe,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(nnodes),
        "--nproc_per_node",
        str(max(1, int(cfg.nproc_per_node))),
        "--max_restarts",
        "0",
    ]
    if nnodes == 1:
        cmd.append("--standalone")
    else:
        cmd.extend(
            [
                "--node_rank",
                str(int(node_rank)),
                "--master_addr",
                str(cfg.master_addr),
                "--master_port",
                str(int(cfg.master_port)),
                "--rdzv_backend",
                str(cfg.rdzv_backend or "c10d"),
                "--rdzv_endpoint",
                f"{cfg.master_addr}:{int(cfg.master_port)}",
                "--rdzv_id",
                str(cfg.rdzv_id or DEFAULT_RDZV_ID),
            ]
        )
    cmd.append(str(script))
    cmd.append("--already-worker")
    if int(cfg.rollout_replicas) != 1:
        cmd.extend(["--rollout-replicas", str(int(cfg.rollout_replicas))])
    cmd.extend(list(train_argv))
    return cmd


def per_node_launch_commands(
    *,
    script: str | Path,
    train_argv: Sequence[str],
    cfg: DistLaunchConfig,
    python_exe: str | None = None,
) -> list[tuple[int, str, list[str]]]:
    hosts = list(cfg.hosts)
    if not hosts:
        hosts = [f"<node{i}>" for i in range(int(cfg.nnodes))]
    if len(hosts) != int(cfg.nnodes):
        raise ValueError(
            f"--hosts has {len(hosts)} entries but --nnodes={cfg.nnodes}"
        )
    rows: list[tuple[int, str, list[str]]] = []
    for rank, host in enumerate(hosts):
        rows.append(
            (
                rank,
                host,
                torchrun_cmd(
                    script=script,
                    train_argv=train_argv,
                    cfg=cfg,
                    node_rank=rank,
                    python_exe=python_exe,
                ),
            )
        )
    return rows


def format_launch_help(rows: Sequence[tuple[int, str, list[str]]]) -> str:
    lines = [
        "Multi-node TRIM RL needs one torchrun process per node.",
        "Run the matching command on each host (shared --out, same argv):",
        "",
    ]
    for rank, host, cmd in rows:
        lines.append(f"# node_rank={rank} host={host}")
        lines.append(" ".join(cmd))
        lines.append("")
    return "\n".join(lines)


def needs_torchrun(cfg: DistLaunchConfig) -> bool:
    if cfg.already_worker or under_torchrun():
        return False
    return int(cfg.nnodes) > 1 or int(cfg.nproc_per_node) > 1
