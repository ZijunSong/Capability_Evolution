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
TRIM_TRAIN_WORKER_ENV = "TRIM_TRAIN_WORKER"
TRIM_EXPECTED_WORLD_SIZE_ENV = "TRIM_EXPECTED_WORLD_SIZE"
TRIM_CUDA_PINNED_ENV = "TRIM_CUDA_PINNED"
TRIM_CUDA_PARENT_VISIBLE_ENV = "TRIM_CUDA_PARENT_VISIBLE"
TRIM_CUDA_PHYSICAL_ENV = "TRIM_CUDA_PHYSICAL"
TRIM_OUTER_RANK_ENV_JSON = "TRIM_OUTER_RANK_ENV_JSON"
DEFAULT_MASTER_PORT = 29500
DEFAULT_DIST_BACKEND = "gloo"
DEFAULT_RDZV_ID = "trim-rl"
DIST_TRAIN_BACKENDS = frozenset({"verl", "fsdp2", "verl_fsdp2", "torch_ddp_lora"})
TORCHRUN_REQUIRED_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
OUTER_STRIP_RANK_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "GROUP_WORLD_SIZE",
    "NODE_RANK",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_USE_AGENT_STORE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
VLLM_CHILD_STRIP_KEYS = OUTER_STRIP_RANK_KEYS + (
    TRIM_TRAIN_WORKER_ENV,
    "PET_MASTER_ADDR",
    "PET_MASTER_PORT",
    "PET_NNODES",
    "PET_NPROC_PER_NODE",
    "PET_NODE_RANK",
)


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
    expected_world_size: int = 1
    outer_rank_env: dict[str, str] = field(default_factory=dict)


_INFO: DistInfo | None = None


def reset_dist_info_for_tests() -> None:
    global _INFO
    _INFO = None


def snapshot_rank_env(env: dict[str, str] | None = None) -> dict[str, str]:
    src = os.environ if env is None else env
    return {k: str(src[k]) for k in OUTER_STRIP_RANK_KEYS if k in src and str(src.get(k) or "") != ""}


def has_complete_torchrun_env(env: dict[str, str] | None = None) -> bool:
    src = os.environ if env is None else env
    return all(str(src.get(k) or "").strip() != "" for k in TORCHRUN_REQUIRED_KEYS)


def is_explicit_train_worker(env: dict[str, str] | None = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get(TRIM_TRAIN_WORKER_ENV) or "").strip() == "1"


def _env_int_or_none(name: str, env: dict[str, str] | None = None) -> int | None:
    src = os.environ if env is None else env
    raw = src.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def pet_nproc_per_node(env: dict[str, str] | None = None) -> str:
    src = os.environ if env is None else env
    return str(src.get("PET_NPROC_PER_NODE") or "").strip()


def pet_nproc_indicates_torchrun(env: dict[str, str] | None = None) -> bool:
    """CloudML / init-pytorch sets ``PET_NPROC_PER_NODE=auto`` on the outer job."""
    pet = pet_nproc_per_node(env)
    return bool(pet) and pet.lower() != "auto"


def torchrun_env_source(env: dict[str, str] | None = None) -> dict[str, Any]:
    src = os.environ if env is None else env
    return {
        "trim_train_worker": is_explicit_train_worker(src),
        "complete_torchrun_env": has_complete_torchrun_env(src),
        "torchelastic_run_id": str(src.get("TORCHELASTIC_RUN_ID") or ""),
        "rank": src.get("RANK"),
        "world_size": src.get("WORLD_SIZE"),
        "local_rank": src.get("LOCAL_RANK"),
        "local_world_size": src.get("LOCAL_WORLD_SIZE"),
        "pet_nproc_per_node": pet_nproc_per_node(src),
        "plan_a_world_size_gt_1": (_env_int_or_none("WORLD_SIZE", src) or 0) > 1,
        "plan_a_pet_nproc_real": pet_nproc_indicates_torchrun(src),
    }


def under_torchrun() -> bool:
    """True only for a real multi-process worker, not CloudML node-level injection.

    CloudML ``framework: pytorch`` / init-pytorch injects ``RANK=0``,
    ``WORLD_SIZE=1`` and ``PET_NPROC_PER_NODE=auto`` even on a single-node
    job. The old check (``RANK`` and ``WORLD_SIZE`` merely present) then
    skipped ``_exec_fsdp2_torchrun``. Plan A: only ``WORLD_SIZE>1`` or a
    non-``auto`` ``PET_NPROC_PER_NODE`` counts as already being under
    torchrun. ``TRIM_TRAIN_WORKER=1`` is our own relaunch mark.
    """
    if is_explicit_train_worker():
        return True
    world = _env_int_or_none("WORLD_SIZE")
    if world is not None and world > 1:
        return True
    if pet_nproc_indicates_torchrun():
        return True
    return False


def is_real_train_worker(cfg: DistLaunchConfig | None = None) -> bool:
    if cfg is not None and cfg.already_worker and has_complete_torchrun_env():
        return True
    return under_torchrun()


def clear_outer_rank_env(env: dict[str, str]) -> dict[str, str]:
    """Strip leftover rank/rendezvous vars from an outer-launcher env copy."""
    saved = snapshot_rank_env(env)
    for key in OUTER_STRIP_RANK_KEYS:
        env.pop(key, None)
    return saved


def isolate_inference_child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Copy env for a TP1 vLLM subprocess: keep the pinned GPU, drop parent rank ids."""
    child = dict(os.environ if env is None else env)
    for key in VLLM_CHILD_STRIP_KEYS:
        child.pop(key, None)
    return child


def expected_world_size_from_env() -> int | None:
    raw = os.environ.get(TRIM_EXPECTED_WORLD_SIZE_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    return max(1, int(raw))


def assert_expected_world_size(info: DistInfo, expected: int | None) -> None:
    if expected is None:
        return
    want = max(1, int(expected))
    got = max(1, int(info.world_size))
    if got != want:
        raise RuntimeError(
            "requested/effective world size mismatch: "
            f"requested={want} effective={got} rank={info.rank} "
            f"local_rank={info.local_rank} local_world_size={info.local_world_size} "
            f"source={torchrun_env_source()}"
        )


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


def _visible_ids(visible: str) -> list[str]:
    return [x.strip() for x in str(visible or "").split(",") if x.strip()]


def pin_local_cuda_device() -> dict[str, Any]:
    """Give each local rank a single visible GPU before torch/vLLM start.

    Idempotent: a second call validates the saved parent/physical mapping and
    returns the same pin. A lone leftover ``CUDA_VISIBLE_DEVICES=1`` is not
    treated as a finished 8-rank bind unless the project pin fields exist.
    After a successful pin the process must use logical ``cuda:0``.
    """
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE") or 1)
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    parent = str(os.environ.get(TRIM_CUDA_PARENT_VISIBLE_ENV) or "")
    physical = str(os.environ.get(TRIM_CUDA_PHYSICAL_ENV) or "")
    already = str(os.environ.get(TRIM_CUDA_PINNED_ENV) or "").strip() == "1"
    payload: dict[str, Any] = {
        "pinned": already,
        "local_rank": local_rank,
        "local_world_size": local_world,
        "cuda_visible_devices": visible,
        "parent_visible": parent,
        "physical_id": physical,
        "logical_device": "cuda:0",
    }
    if already:
        current_ids = _visible_ids(visible)
        if physical and current_ids and current_ids != [physical]:
            raise RuntimeError(
                "CUDA pin mismatch on repeat bind: "
                f"CUDA_VISIBLE_DEVICES={visible!r} saved_physical={physical!r}"
            )
        payload["cuda_visible_devices"] = physical or (current_ids[0] if current_ids else visible)
        return payload
    if local_world <= 1:
        return payload
    if visible and visible not in {"-1", "none", "None"}:
        ids = _visible_ids(visible)
        if len(ids) == 1:
            raise RuntimeError(
                "refusing to treat a single-element CUDA_VISIBLE_DEVICES as an 8-rank pin "
                f"without {TRIM_CUDA_PINNED_ENV}=1 "
                f"(LOCAL_RANK={local_rank} LOCAL_WORLD_SIZE={local_world} visible={visible!r}). "
                "The outer launcher must pass the full parent GPU list."
            )
        if local_rank >= len(ids):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but CUDA_VISIBLE_DEVICES has {len(ids)} ids: {visible!r}"
            )
        chosen = ids[local_rank]
        parent_visible = ",".join(ids)
    else:
        chosen = str(local_rank)
        parent_visible = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    os.environ[TRIM_CUDA_PINNED_ENV] = "1"
    os.environ[TRIM_CUDA_PARENT_VISIBLE_ENV] = parent_visible
    os.environ[TRIM_CUDA_PHYSICAL_ENV] = chosen
    payload.update(
        {
            "pinned": True,
            "cuda_visible_devices": chosen,
            "parent_visible": parent_visible,
            "physical_id": chosen,
        }
    )
    return payload


def collect_rank_probe() -> dict[str, Any]:
    import socket

    info = dist_info()
    logical = "cpu"
    uuid = None
    name = None
    if str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip() not in {"", "-1", "none", "None"}:
        logical = "cuda:0"
    try:
        import torch

        if torch.cuda.is_available():
            logical = "cuda:0"
            props = torch.cuda.get_device_properties(0)
            name = str(getattr(props, "name", "") or "")
            raw_uuid = getattr(props, "uuid", None)
            uuid = str(raw_uuid) if raw_uuid is not None else None
    except Exception:
        pass
    return {
        "rank": int(info.rank),
        "local_rank": int(info.local_rank),
        "world_size": int(info.world_size),
        "pid": int(os.getpid()),
        "host": socket.gethostname(),
        "logical_device": logical,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        "physical_id": str(os.environ.get(TRIM_CUDA_PHYSICAL_ENV) or ""),
        "gpu_uuid": uuid,
        "gpu_name": name,
        "source": torchrun_env_source(),
    }


def assert_unique_local_gpu_uuids(probes: Sequence[dict[str, Any]]) -> None:
    by_host: dict[str, list[dict[str, Any]]] = {}
    for row in probes:
        by_host.setdefault(str(row.get("host") or ""), []).append(dict(row))
    for host, rows in by_host.items():
        if len(rows) <= 1:
            continue
        uuids = [str(r.get("gpu_uuid") or "") for r in rows]
        if any(not u or u == "None" for u in uuids):
            physical = [str(r.get("physical_id") or r.get("cuda_visible_devices") or "") for r in rows]
            if len(set(physical)) != len(physical):
                raise RuntimeError(
                    f"local GPU bind is not unique on host={host}: {rows}"
                )
            continue
        if len(set(uuids)) != len(uuids):
            raise RuntimeError(f"duplicate GPU UUID on host={host}: {rows}")


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


def _safe_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(tag))


def resolve_rollout_shard_dir(*, out: Path, run_id: str, single_node: bool) -> Path:
    """Job-local scratch for rollout IPC.

    Single-node jobs use a run-scoped directory under ``/tmp`` (or
    ``TRIM_ROLLOUT_SHARD_DIR``). The directory includes ``run_id`` so two jobs
    on one machine do not share ``bN`` files.
    """
    safe = _safe_tag(run_id) or "run"
    override = os.environ.get("TRIM_ROLLOUT_SHARD_DIR")
    if override:
        path = Path(override) / safe / "rollout_shards"
    elif single_node:
        path = Path("/tmp") / "trim" / safe / "rollout_shards"
    else:
        path = Path(out) / "tmp" / "rollout_shards" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_policy_pickle(path: Path, items: Sequence[Any], *, policy_version: str | None) -> None:
    """Atomically publish a shard. A policy version is stored when provided."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if policy_version is None:
        payload: Any = list(items)
    else:
        payload = {"policy_version": str(policy_version), "items": list(items)}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def read_policy_pickle(path: Path, *, policy_version: str | None) -> list[Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if policy_version is None:
        if isinstance(payload, dict) and "items" in payload:
            return list(payload["items"])
        return list(payload)
    found = payload.get("policy_version") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or str(found or "") != str(policy_version):
        raise RuntimeError(
            f"rollout shard {path} policy_version={found!r} does not match expected {policy_version!r}"
        )
    return list(payload.get("items") or [])


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def gather_sharded_objects(
    local: Sequence[T],
    *,
    shard_dir: Path,
    tag: str,
    policy_version: str | None = None,
) -> list[T]:
    """Write per-rank pickles to shared disk, rank 0 interleaves them.

    Trajectories can be large; file gather avoids NCCL object-size limits.
    ``policy_version`` rejects a shard left by another policy in the same directory.
    """
    info = dist_info()
    items = list(local)
    if info.world_size <= 1:
        return items
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = _safe_tag(tag)
    path = shard_dir / f"{safe_tag}.rank{int(info.rank):04d}.pkl"
    write_policy_pickle(path, items, policy_version=policy_version)
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
            shards.append(read_policy_pickle(shard_path, policy_version=policy_version))
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


def all_gather_via_disk(
    local: Sequence[T],
    *,
    shard_dir: Path,
    tag: str,
    policy_version: str | None = None,
) -> list[T]:
    """Like gather_sharded_objects, then share the merged list with every rank."""
    info = dist_info()
    merged = gather_sharded_objects(local, shard_dir=shard_dir, tag=tag, policy_version=policy_version)
    if info.world_size <= 1:
        return merged
    blob_path = Path(shard_dir) / f"{_safe_tag(tag)}.all.pkl"
    if info.is_coordinator:
        write_policy_pickle(blob_path, merged, policy_version=policy_version)
    barrier()
    if not info.is_coordinator:
        merged = read_policy_pickle(blob_path, policy_version=policy_version)
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
    parser.add_argument(
        "--expected-world-size",
        dest="expected_world_size",
        type=int,
        default=None,
        help="Requested training world size. Workers abort if torch.distributed disagrees.",
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
    expected = cfg_ns.expected_world_size
    if expected is None:
        expected = expected_world_size_from_env()
    if expected is None:
        expected = max(1, int(nnodes)) * max(1, int(nproc))
    already = bool(cfg_ns.already_worker) or is_real_train_worker()
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
        already_worker=already,
        expected_world_size=max(1, int(expected)),
        outer_rank_env=snapshot_rank_env(),
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
    cmd.extend(["--expected-world-size", str(max(1, int(cfg.expected_world_size or cfg.nnodes * cfg.nproc_per_node)))])
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
    return int(cfg.nnodes) > 1 or int(cfg.nproc_per_node) > 1 or int(cfg.expected_world_size) > 1
