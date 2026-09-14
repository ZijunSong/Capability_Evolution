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


def load_rng_state(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    py = payload.get("python")
    if isinstance(py, list) and len(py) >= 3:
        random.setstate((py[0], tuple(py[1]), py[2]))
    torch_state = payload.get("torch")
    if torch_state:
        torch.random.set_rng_state(torch.tensor(torch_state, dtype=torch.uint8))
    cuda_states = payload.get("cuda") or []
    if torch.cuda.is_available():
        for i, state in enumerate(cuda_states):
            if i < torch.cuda.device_count() and state:
                torch.cuda.set_rng_state(torch.tensor(state, dtype=torch.uint8), i)
    return True


def training_output_occupied(out: Path) -> bool:
    if (out / "TRAIN_SUMMARY.json").is_file() or (out / "RUN_COMPLETE").is_file():
        return True
    ckpt = out / "checkpoints"
    if ckpt.is_dir() and any(ckpt.iterdir()):
        return True
    if (out / "optimizer_latest.pt").is_file():
        return True
    return False


def find_latest_checkpoint(cell_ckpt_dir: Path) -> Path | None:
    latest = cell_ckpt_dir / "latest"
    if latest.exists() and (latest / "STEP_COMPLETE").is_file():
        return latest.resolve() if latest.is_symlink() else latest
    if not cell_ckpt_dir.is_dir():
        return None
    steps = sorted(
        [p for p in cell_ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: p.name,
    )
    for path in reversed(steps):
        if (path / "STEP_COMPLETE").is_file():
            return path
    return None


def load_training_resume(cell_ckpt_dir: Path) -> dict[str, Any] | None:
    ckpt = find_latest_checkpoint(cell_ckpt_dir)
    if ckpt is None:
        return None
    manifest_path = ckpt / "STEP_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    sampler_path = ckpt / "sampler.json"
    sampler = json.loads(sampler_path.read_text(encoding="utf-8")) if sampler_path.is_file() else {}
    return {
        "checkpoint_dir": ckpt,
        "adapter_dir": ckpt / "adapter",
        "optimizer_path": ckpt / "optimizer.pt",
        "rng_path": ckpt / "rng.json",
        "sampler_state": sampler,
        "manifest": manifest,
        "step": int(manifest.get("step") or sampler.get("global_optimizer_step") or 0),
        "source_policy_version": str(manifest.get("source_policy_version") or ""),
        "updated_policy_version": str(
            manifest.get("updated_policy_version") or manifest.get("policy_version") or "v0"
        ),
    }


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
