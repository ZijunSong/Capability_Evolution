#!/usr/bin/env python3
"""Round 7 environment and asset preflight snapshot."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round7/preflight"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=_REPO, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return str(e)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = {
        "base_model": "/data/ppnm/models/Qwen2.5-7B-Instruct",
        "manifest": str(_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json"),
        "valid522": str(_REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl"),
        "harness_config": str(_REPO / "harness/configs/modules_minimal_v2.yaml"),
    }
    for seed in (42, 43, 44):
        assets[f"o7_seed{seed}"] = str(_REPO / f"outputs/scope_round5/merged/o7_r64_seed{seed}")

    snap = {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "branch", "--show-current"]),
        "git_diff_stat": _run(["git", "diff", "--stat"]),
        "nvidia_smi": _run(["nvidia-smi", "-L"]),
        "gpu_memory": _run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader"]),
        "python": _run(["python", "--version"]),
        "pytorch": _run(["python", "-c", "import torch; print(torch.__version__)"]),
        "transformers": _run(["python", "-c", "import transformers; print(transformers.__version__)"]),
        "peft": _run(["python", "-c", "import peft; print(peft.__version__)"]),
        "vllm": _run(["python", "-c", "import vllm; print(vllm.__version__)"]),
        "disk_free": _run(["df", "-h", str(_REPO)]),
        "assets_exist": {k: Path(v).exists() for k, v in assets.items()},
    }
    (OUT / "environment_snapshot.json").write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()
