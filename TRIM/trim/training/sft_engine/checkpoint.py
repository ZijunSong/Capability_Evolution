"""v2 checkpoints. Adapter first, then optimizer, then an atomic commit."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.checkpoint import assert_trainable_signature, capture_rng, restore_rng, trainable_signature
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_file, sha256_tree
from trim.training.sft_engine.contract import require_exact_resume


def _torch():
    import torch

    return torch


def latest_v2(root: Path) -> Path | None:
    root = Path(root)
    if not root.is_dir():
        return None
    found = [path for path in root.iterdir() if path.is_dir() and (path / "COMMITTED.json").is_file()]
    if not found:
        return None
    return sorted(found)[-1]


def find_selection(root: Path, selection_digest: str) -> Path | None:
    root = Path(root)
    if not root.is_dir() or not selection_digest:
        return None
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        manifest_path = path / "SFT_CHECKPOINT.json"
        committed = path / "COMMITTED.json"
        if not manifest_path.is_file() or not committed.is_file():
            continue
        manifest = read_json(manifest_path)
        if str(manifest.get("selection_digest") or "") == selection_digest:
            return path
    return None


def publish_v2_checkpoint(
    acc: Any,
    directory: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    contract: dict[str, Any],
    optimizer_step: int,
    policy_version: str,
    source_digest: str,
    rollout_adapter_digest: str,
    sampler: dict[str, Any],
    selection_digest: str = "",
    pending_round: int | None = None,
) -> Path:
    """Rank 0 writes the adapter and optimizer. Every rank writes its RNG.

    ``COMMITTED.json`` is renamed into place only after the checksum file is
    complete. Other ranks observe the broadcast status and then continue.
    """
    import torch.distributed as dist

    torch = _torch()
    directory = Path(directory)
    rank = int(acc.process_index)
    world = int(acc.num_processes)
    staging = directory.parent / f".tmp-{directory.name}"
    rng_root = directory.parent / f".tmp-{directory.name}-rng"
    if rank == 0:
        if staging.exists():
            shutil.rmtree(staging)
        if rng_root.exists():
            shutil.rmtree(rng_root)
        rng_root.mkdir(parents=True)
    _barrier(acc)
    torch.save(capture_rng(), rng_root / f"rank-{rank:05d}.pt")
    _barrier(acc)
    status = 1
    error = ""
    if rank == 0:
        try:
            _write_rank0(
                staging,
                rng_root,
                world=world,
                model=acc.unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                contract=contract,
                optimizer_step=optimizer_step,
                policy_version=policy_version,
                source_digest=source_digest,
                rollout_adapter_digest=rollout_adapter_digest,
                sampler=sampler,
                selection_digest=selection_digest,
                pending_round=pending_round,
            )
            if directory.exists():
                raise FileExistsError(f"checkpoint already exists: {directory}")
            staging.rename(directory)
        except Exception as exc:
            status = 0
            error = f"{type(exc).__name__}: {exc}"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    flag = torch.tensor([status], dtype=torch.int32, device=acc.device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(flag, src=0)
    _barrier(acc)
    if rank == 0 and rng_root.exists():
        shutil.rmtree(rng_root, ignore_errors=True)
    if int(flag.item()) != 1:
        raise RuntimeError(error or "checkpoint commit failed")
    if rank == 0:
        try:
            atomic_write_json(directory.parent.parent / "latest.json", {"checkpoint": str(directory)})
        except Exception:
            pass
    _barrier(acc)
    return directory


def _write_rank0(
    staging: Path,
    rng_root: Path,
    *,
    world: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    contract: dict[str, Any],
    optimizer_step: int,
    policy_version: str,
    source_digest: str,
    rollout_adapter_digest: str,
    sampler: dict[str, Any],
    selection_digest: str,
    pending_round: int | None,
) -> None:
    torch = _torch()
    staging.mkdir(parents=True)
    adapter = staging / "adapter"
    adapter.mkdir()
    if not hasattr(model, "save_pretrained"):
        raise TypeError("v2 checkpoints require a PEFT model with save_pretrained")
    model.save_pretrained(adapter)
    adapter_digest = sha256_tree(adapter)
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    torch.save(None if scheduler is None else scheduler.state_dict(), staging / "scheduler.pt")
    rng_dir = staging / "rng"
    rng_dir.mkdir()
    for rank in range(world):
        source = rng_root / f"rank-{rank:05d}.pt"
        if not source.is_file():
            raise FileNotFoundError(f"missing rng for rank {rank}")
        shutil.copy2(source, rng_dir / source.name)
    atomic_write_json(staging / "sampler.json", sampler)
    atomic_write_json(
        staging / "SFT_CHECKPOINT.json",
        {
            "schema_version": "sft_checkpoint_v2",
            "contract": contract,
            "optimizer_step": int(optimizer_step),
            "policy_version": policy_version,
            "adapter_digest": adapter_digest,
            "source_digest": source_digest,
            "rollout_adapter_digest": rollout_adapter_digest,
            "selection_digest": selection_digest,
            "pending_round": pending_round,
            "trainable": trainable_signature(model),
        },
    )
    names = ["optimizer.pt", "scheduler.pt", "sampler.json", "SFT_CHECKPOINT.json"]
    names.extend(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file() and path.name != "COMMITTED.json")
    # The list above includes files added while walking, so freeze it first.
    names = sorted(set(names))
    atomic_write_json(staging / "COMMITTED.json", {"files": {name: sha256_file(staging / name) for name in names}})


def load_v2_training_state(
    directory: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    contract: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    """Load optimizer, scheduler, and this rank's RNG.

    The caller must already have built the base, loaded this adapter, checked
    the trainable signature, created the optimizer, and wrapped the model.
    """
    directory = Path(directory)
    committed_path = directory / "COMMITTED.json"
    manifest_path = directory / "SFT_CHECKPOINT.json"
    if not committed_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"refusing uncommitted checkpoint: {directory}")
    recorded = read_json(committed_path)["files"]
    for name, digest in recorded.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    manifest = read_json(manifest_path)
    require_exact_resume(manifest["contract"], contract)
    assert_trainable_signature(model, list(manifest["trainable"]))
    if sha256_tree(directory / "adapter") != manifest.get("adapter_digest"):
        raise ValueError("adapter directory digest does not match the checkpoint")
    torch = _torch()
    optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False))
    sched_state = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=False)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)
    rng_path = directory / "rng" / f"rank-{int(rank):05d}.pt"
    if not rng_path.is_file():
        raise ValueError(f"checkpoint has no RNG for rank {rank}")
    restore_rng(torch.load(rng_path, map_location="cpu", weights_only=False))
    manifest["sampler"] = read_json(directory / "sampler.json")
    return manifest


def _barrier(acc: Any) -> None:
    wait = getattr(acc, "wait_for_everyone", None)
    if callable(wait):
        wait()
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
