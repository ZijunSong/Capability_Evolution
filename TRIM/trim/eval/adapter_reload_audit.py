"""Save / reload audit for four-cell adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(adapter_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not adapter_dir.is_dir():
        return out
    for path in sorted(adapter_dir.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(adapter_dir))] = sha256_file(path)
    return out


def audit_saved_adapter(adapter_dir: Path, *, cell: str) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    inventory = file_inventory(adapter_dir)
    return {
        "cell": cell,
        "adapter_dir": str(adapter_dir),
        "exists": adapter_dir.is_dir(),
        "missing_required": missing,
        "n_files": len(inventory),
        "sha256": inventory,
        "reload_ready": adapter_dir.is_dir() and not missing,
    }


def remap_lora_state(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        ): value
        for key, value in raw.items()
    }


def reload_adapter(base_model: Any, adapter_dir: Path) -> tuple[Any, str]:
    """Load a PEFT adapter onto an already-constructed base model."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter missing: {adapter_dir}")
    try:
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        return model, "peft_model_from_pretrained"
    except Exception:
        cfg_path = adapter_dir / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        lc = LoraConfig(
            task_type=cfg.get("task_type", "CAUSAL_LM"),
            r=int(cfg.get("r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=list(cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=cfg.get("bias", "none"),
        )
        model = get_peft_model(base_model, lc)
        weights = remap_lora_state(load_file(str(adapter_dir / "adapter_model.safetensors")))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        lora_missing = [x for x in missing if "lora_" in x]
        lora_unexpected = [x for x in unexpected if "lora_" in x]
        if lora_missing or lora_unexpected:
            raise RuntimeError(f"manual adapter reload mismatch missing={lora_missing} unexpected={lora_unexpected}")
        return model, "manual_safetensors_state_dict"


def write_reload_audit(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "n_cells": len(rows),
        "all_reload_ready": all(bool(r.get("reload_ready")) for r in rows if r.get("cell") != "before"),
        "cells": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
