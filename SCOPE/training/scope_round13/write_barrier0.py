#!/usr/bin/env python3
"""Write OLD_HOLDOUT_RETIREMENT.md + ENVIRONMENT_SNAPSHOT.txt."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round13"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "OLD_HOLDOUT_RETIREMENT.md"
    md.write_text(
        """# OLD HOLDOUT RETIREMENT (Round13)

From Round13 onward, the following datasets are **historical diagnostic only**:

- Round8/Round9 `offline_valid`
- Round9/Round10/Round11/Round12 `base_live`
- `artifacts/datasets/round2_audit_100q/query_manifest.json`

They MUST NOT be used for:

- threshold selection
- hyperparameter selection
- variant ranking
- early stopping
- final success claim

Fresh Round13 splits (`R13_TRAIN200` / `VALID100` / `TEST100` / `SMOKE20` / `FINAL100`)
are the only authorized model-selection and evaluation surfaces for this round.
""",
        encoding="utf-8",
    )

    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=_REPO, text=True
        ).strip(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "bishop"),
        "base_model": "/data/ppnm/models/Qwen2.5-7B-Instruct",
        "collect_model": str(
            _REPO / "outputs/scope_round11/phase_b/factorized_full_stage1_seed42/merged"
        ),
        "n_gpus": 8,
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        snap["torch_error"] = str(exc)

    (OUT / "ENVIRONMENT_SNAPSHOT.txt").write_text(
        json.dumps(snap, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {md} and ENVIRONMENT_SNAPSHOT.txt")


if __name__ == "__main__":
    main()
