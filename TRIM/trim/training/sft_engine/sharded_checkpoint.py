"""FSDP2 checkpoints: Distributed Checkpoint for train state, aggregated LoRA for inference.

Separate from the DDP ``checkpoint.py`` on purpose. The frozen BF16 base is
never rewritten into checkpoints; the run contract pins the base identity and
content digests cover only what is stored here. Every rank participates in the
DCP save/load and in the LoRA ``full_tensor()`` aggregation — only the final
directory publication is rank-0 work.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.checkpoint import capture_rng, restore_rng
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_file, sha256_tree
from trim.training.sft_engine.contract import require_exact_resume

SCHEMA_VERSION = "sft_checkpoint_v3"


def _torch():
    import torch

    return torch


def raw_optimizer(optimizer: Any) -> Any:
    try:
        from accelerate.optimizer import AcceleratedOptimizer
    except Exception:
        return optimizer
    while isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    return optimizer


def save_sharded_train_state(model: Any, optimizer: Any, path: Path) -> None:
    """All-rank DCP write of trainable params + their optimizer state."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False)
    model_sd, optim_sd = get_state_dict(model, raw_optimizer(optimizer), options=options)
    dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(path))


def load_sharded_train_state(model: Any, optimizer: Any, path: Path) -> dict[str, Any]:
    """All-rank DCP load. Missing trainable keys or extra keys are fatal."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False)
    model_sd, optim_sd = get_state_dict(model, raw_optimizer(optimizer), options=options)
    state = {"model": model_sd, "optimizer": optim_sd}
    dcp.load(state, checkpoint_id=str(path))
    result = set_state_dict(
        model,
        raw_optimizer(optimizer),
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    bad_missing = trainable.intersection(result.missing_keys)
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f"sharded restore mismatch: missing={sorted(bad_missing)} extra={sorted(result.unexpected_keys)}")
    return {"missing_keys": sorted(result.missing_keys), "unexpected_keys": sorted(result.unexpected_keys)}


def export_lora_fsdp2(acc: Any, model: Any, destination: Path, *, exist_ok: bool = False) -> None:
    """Aggregate the trainable LoRA into a plain PEFT adapter directory.

    Do not aggregate the base to export an adapter: only trainable parameters
    enter the collective. ``full_tensor()`` must run on every rank even when
    rank 0 alone writes the files.
    """
    from pathlib import Path as _Path

    import torch
    from torch.distributed.tensor import DTensor

    core = acc.unwrap_model(model)
    pairs = sorted((name, param) for name, param in core.named_parameters() if param.requires_grad)
    if not pairs or any("lora_" not in name for name, _ in pairs):
        raise RuntimeError("adapter export only supports the common LoRA contract")
    full_sd: dict[str, Any] = {}
    for name, param in pairs:
        with torch.no_grad():
            tensor = param.detach()
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()  # EVERY rank enters this collective.
            if acc.is_main_process:
                full_sd[name] = tensor.cpu().contiguous()
            del tensor
    if acc.is_main_process:
        destination = _Path(destination)
        if destination.exists():
            if not exist_ok:
                raise FileExistsError(f"adapter export target already exists: {destination}")
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        try:
            core.save_pretrained(
                destination,
                state_dict=full_sd,
                safe_serialization=True,
                save_embedding_layers=False,
            )
        except TypeError:
            # Older PEFT without save_embedding_layers support.
            core.save_pretrained(destination, state_dict=full_sd, safe_serialization=True)
    acc.wait_for_everyone()


def publish_sharded_checkpoint(
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
    """All-rank sharded save, then rank-0 atomic publish.

    Mirrors the v2 DDP protocol: staging directory, per-rank RNG, checksummed
    COMMITTED.json, rename into place, then the ``latest.json`` pointer.
    """
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
            staging.mkdir(parents=True)
            (staging / "dcp").mkdir()
        except Exception as exc:
            status = 0
            error = f"{type(exc).__name__}: {exc}"
    _broadcast_status(acc, status, error)
    # Collective phase: DCP write and LoRA aggregation, every rank.
    save_sharded_train_state(model, optimizer, staging / "dcp")
    export_lora_fsdp2(acc, model, staging / "adapter")
    if rank == 0:
        try:
            _finalize_rank0(
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
    _broadcast_status(acc, status, error)
    if rank == 0 and rng_root.exists():
        shutil.rmtree(rng_root, ignore_errors=True)
    if rank == 0:
        try:
            atomic_write_json(directory.parent.parent / "latest.json", {"checkpoint": str(directory)})
        except Exception:
            pass
    _barrier(acc)
    return directory


def _finalize_rank0(
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
    adapter_digest = sha256_tree(staging / "adapter")
    torch.save(None if scheduler is None else scheduler.state_dict(), staging / "scheduler.pt")
    rng_dir = staging / "rng"
    rng_dir.mkdir()
    for rank in range(world):
        source = rng_root / f"rank-{rank:05d}.pt"
        if not source.is_file():
            raise FileNotFoundError(f"missing rng for rank {rank}")
        shutil.copy2(source, rng_dir / source.name)
    atomic_write_json(staging / "sampler.json", sampler)
    from trim.training.sft_ablation.checkpoint import trainable_signature

    atomic_write_json(
        staging / "SFT_CHECKPOINT.json",
        {
            "schema_version": SCHEMA_VERSION,
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
    names = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != "COMMITTED.json"
    )
    atomic_write_json(staging / "COMMITTED.json", {"files": {name: sha256_file(staging / name) for name in names}})


def load_sharded_training_state(
    directory: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    contract: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    """Verify the committed artifact, restore optimizer state and this rank's RNG.

    The caller must already have prepared the model/optimizer and loaded this
    checkpoint's adapter into the base (or started from the same init adapter
    before DCP overwrite — see the run contract for the accepted resume mode).
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
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA_VERSION}, got {manifest.get('schema_version')}")
    require_exact_resume(manifest["contract"], contract)
    from trim.training.sft_ablation.checkpoint import assert_trainable_signature

    assert_trainable_signature(model, list(manifest["trainable"]))
    if sha256_tree(directory / "adapter") != manifest.get("adapter_digest"):
        raise ValueError("adapter directory digest does not match the checkpoint")
    load_sharded_train_state(model, optimizer, directory / "dcp")
    torch = _torch()
    sched_state = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=False)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)
    rng_path = directory / "rng" / f"rank-{int(rank):05d}.pt"
    if not rng_path.is_file():
        raise ValueError(f"checkpoint has no RNG for rank {rank}")
    restore_rng(torch.load(rng_path, map_location="cpu", weights_only=False))
    manifest["sampler"] = read_json(directory / "sampler.json")
    return manifest


def _broadcast_status(acc: Any, status: int, error: str) -> None:
    """Share rank 0's staging outcome; every rank aborts if rank 0 failed.

    Non-rank-0 exceptions kill the world through the torchrun launcher instead
    of trying to reach consensus on an already-broken collective.
    """
    import torch
    import torch.distributed as dist

    flag = torch.tensor([int(status)], dtype=torch.int32, device=acc.device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(flag, src=0)
    _barrier(acc)
    if int(flag.item()) != 1:
        raise RuntimeError(error or "rank 0 failed while publishing a sharded checkpoint")


def _barrier(acc: Any) -> None:
    wait = getattr(acc, "wait_for_everyone", None)
    if callable(wait):
        wait()
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
