#!/usr/bin/env python3
"""SCAPE preflight — env, CUDA, model, canonical imports (no SCOPE train_opd)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "scape.state.snapshot",
    "scape.rendering.dual_view",
    "scape.training.tool_mask",
    "scape.training.tool_opd",
    "scape.training.teacher",
    "scape.training.hf_tool_opd",
    "scape.collection.same_state",
]

FORBIDDEN_IMPORTS = [
    "training.train_opd",
    "training.smoke_opd_vllm_hf",
]


def _nvidia_smi() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_used": parts[2],
                        "memory_total": parts[3],
                    }
                )
        return {"ok": True, "gpus": gpus}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/data/ppnm/models/Qwen2.5-7B-Instruct"),
    )
    ap.add_argument("--require-cuda", action="store_true", default=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {"ok": True, "checks": {}}

    # Python path
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Modules
    mod_ok = True
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            report["checks"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["checks"][name] = f"FAIL: {exc}"
            mod_ok = False
            report["ok"] = False

    # Forbid accidental SCOPE canonical imports being required
    for name in FORBIDDEN_IMPORTS:
        try:
            importlib.import_module(name)
            report["checks"][f"forbidden:{name}"] = "PRESENT (warn — must not be canonical)"
        except Exception:
            report["checks"][f"forbidden:{name}"] = "absent_ok"

    # CUDA / torch
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        report["checks"]["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_ok,
            "device_count": int(torch.cuda.device_count()) if cuda_ok else 0,
        }
        if args.require_cuda and not cuda_ok:
            report["ok"] = False
    except Exception as exc:  # noqa: BLE001
        report["checks"]["torch"] = f"FAIL: {exc}"
        report["ok"] = False
        cuda_ok = False

    smi = _nvidia_smi()
    report["checks"]["nvidia_smi"] = smi
    if args.require_cuda and not smi.get("ok"):
        report["ok"] = False

    model_path = Path(args.model_path)
    report["checks"]["model_path"] = {
        "path": str(model_path),
        "exists": model_path.exists(),
    }
    if not model_path.exists():
        report["ok"] = False

    # Loss path distinctness
    try:
        from scape.training.hf_tool_opd import assert_loss_paths_distinct

        dist = assert_loss_paths_distinct()
        report["checks"]["loss_paths_distinct"] = dist
        if not dist.get("distinct"):
            report["ok"] = False
            mod_ok = False
    except Exception as exc:  # noqa: BLE001
        report["checks"]["loss_paths_distinct"] = f"FAIL: {exc}"
        report["ok"] = False

    report["legacy_scope_path_used"] = False
    report["repo"] = str(repo)
    report["modules_ok"] = mod_ok

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
