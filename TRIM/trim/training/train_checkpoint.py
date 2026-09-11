"""Per-step training checkpoints with optimizer and sampler state."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch


def optimizer_state_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.detach().cpu()
        elif isinstance(val, dict):
            out[key] = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in val.items()
            }
        else:
            out[key] = val
    return out


def param_group_signature(params) -> str:
    import hashlib

    parts = []
    for p in params:
        if p.requires_grad:
            parts.append(f"{p.shape}:{p.dtype}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def save_optimizer_bundle(backend: Any, path: Path) -> dict[str, Any]:
    opt = getattr(backend, "optimizer", None)
    if opt is None:
        return {"saved": False}
    trainable = [p for p in backend.model.parameters() if p.requires_grad]
    bundle = {
        "saved": True,
        "signature": param_group_signature(trainable),
        "state_dict": optimizer_state_to_cpu(opt.state_dict()),
        "param_groups": [
            {k: v for k, v in g.items() if k != "params"} for g in opt.param_groups
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return {"saved": True, "path": str(path), "signature": bundle["signature"]}


def load_optimizer_bundle(backend: Any, path: Path) -> bool:
    if not path.is_file():
        return False
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    trainable = [p for p in backend.model.parameters() if p.requires_grad]
    sig = param_group_signature(trainable)
    if str(bundle.get("signature")) != sig:
        raise RuntimeError(
            f"optimizer signature mismatch: checkpoint={bundle.get('signature')} live={sig}"
        )
    if backend.optimizer is None:
        lr = float(bundle["param_groups"][0].get("lr", 1e-5)) if bundle.get("param_groups") else 1e-5
        backend.optimizer = torch.optim.AdamW(trainable, lr=lr)
        for i, g in enumerate(bundle.get("param_groups") or []):
            if i < len(backend.optimizer.param_groups):
                backend.optimizer.param_groups[i].update(
                    {k: v for k, v in g.items() if k != "params"}
                )
    state = bundle["state_dict"]
    device = backend._device
    for key, val in state.items():
        if isinstance(val, torch.Tensor):
            state[key] = val.to(device)
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, torch.Tensor):
                    val[k2] = v2.to(device)
    backend.optimizer.load_state_dict(state)
    return True


def save_rng_state(path: Path) -> None:
    payload = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state().tolist(),
        "cuda": [torch.cuda.get_rng_state(i).tolist() for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def publish_step_checkpoint(
    tmp_dir: Path,
    final_dir: Path,
    *,
    manifest: dict[str, Any],
) -> None:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "STEP_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (tmp_dir / "STEP_COMPLETE").write_text("1\n", encoding="utf-8")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    latest = final_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink(missing_ok=True)
    os.symlink(final_dir.name, latest)


def append_metrics_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
