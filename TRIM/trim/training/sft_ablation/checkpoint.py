"""Strict SFT checkpoints. A committed directory is the only resume source."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.contracts import require_same_identity
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_file


def _torch():
    import torch

    return torch


def capture_rng() -> dict[str, Any]:
    torch = _torch()
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        payload["numpy"] = np.random.get_state()
    except Exception:
        payload["numpy"] = None
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: dict[str, Any]) -> None:
    torch = _torch()
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch_cpu"])
    if payload.get("numpy") is not None:
        import numpy as np

        np.random.set_state(payload["numpy"])
    if payload.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def trainable_signature(model: Any) -> list[dict[str, str]]:
    rows = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            rows.append({"name": name, "shape": list(param.shape), "dtype": str(param.dtype)})
    return rows


def assert_trainable_signature(model: Any, saved: list[dict[str, str]]) -> None:
    live = trainable_signature(model)
    if live != saved:
        raise ValueError("optimizer parameter name/shape/dtype mismatch")


def _checksums(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def publish_checkpoint(
    directory: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    identity: dict[str, Any],
    optimizer_step: int,
    next_round_id: int,
    policy_version: str,
    adapter_digest: str,
    source_digest: str,
    sampler: dict[str, Any],
) -> Path:
    """Write a temp directory, then rename it. COMMITTED.json is last."""
    import os

    torch = _torch()
    directory = Path(directory)
    tmp = directory.parent / f".tmp-{directory.name}"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    adapter = tmp / "adapter"
    adapter.mkdir()
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter)
    else:
        torch.save(model.state_dict(), adapter / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(None if scheduler is None else scheduler.state_dict(), tmp / "scheduler.pt")
    torch.save(capture_rng(), tmp / "rng.pt")
    atomic_write_json(tmp / "sampler.json", sampler)
    signature = trainable_signature(model)
    atomic_write_json(
        tmp / "SFT_CHECKPOINT.json",
        {
            "immutable": identity,
            "optimizer_step": int(optimizer_step),
            "next_round_id": int(next_round_id),
            "policy_version": policy_version,
            "adapter_digest": adapter_digest,
            "source_digest": source_digest,
            "trainable": signature,
        },
    )
    names = [
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
        "sampler.json",
        "SFT_CHECKPOINT.json",
    ]
    names.extend(path.relative_to(tmp).as_posix() for path in adapter.rglob("*") if path.is_file())
    atomic_write_json(tmp / "COMMITTED.json", {"files": _checksums(tmp, names)})
    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: {directory}")
    os.rename(tmp, directory)
    return directory


def load_committed(directory: Path, *, model: Any, optimizer: Any, scheduler: Any, identity: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    committed = directory / "COMMITTED.json"
    manifest_path = directory / "SFT_CHECKPOINT.json"
    if not committed.is_file() or not manifest_path.is_file():
        raise ValueError(f"refusing uncommitted checkpoint: {directory}")
    recorded = read_json(committed)["files"]
    for name, digest in recorded.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    manifest = read_json(manifest_path)
    require_same_identity(manifest["immutable"], identity)
    assert_trainable_signature(model, list(manifest["trainable"]))
    torch = _torch()
    if (directory / "adapter" / "pytorch_model.bin").is_file() and not hasattr(model, "load_adapter"):
        model.load_state_dict(torch.load(directory / "adapter" / "pytorch_model.bin", map_location="cpu", weights_only=False))
    elif hasattr(model, "load_adapter"):
        model.load_adapter(str(directory / "adapter"), adapter_name="default")
    optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False))
    sched_state = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=False)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)
    restore_rng(torch.load(directory / "rng.pt", map_location="cpu", weights_only=False))
    manifest["sampler"] = read_json(directory / "sampler.json")
    return manifest


def latest_committed(root: Path) -> Path | None:
    root = Path(root)
    if not root.is_dir():
        return None
    found = [path for path in root.iterdir() if path.is_dir() and (path / "COMMITTED.json").is_file()]
    if not found:
        return None
    return sorted(found)[-1]


def write_latest(out: Path, checkpoint: Path) -> None:
    atomic_write_json(out / "latest.json", {"checkpoint": str(checkpoint)})
