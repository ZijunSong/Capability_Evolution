#!/usr/bin/env python3
"""Queue SCOPE config ablations (verification / evidence / endorse-only / dual / weights)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ABLATIONS = {
    "dual_mode": "configs/scope/dual_mode.yaml",
    "verification_only": "configs/scope/verification_only.yaml",
    "evidence_only": "configs/scope/evidence_only.yaml",
    "endorse_only": "configs/scope/endorse_only.yaml",
    "fixed_weight": "configs/scope/fixed_weight.yaml",
    "adaptive_weight": "configs/scope/adaptive_weight.yaml",
    "minimal_runtime": "configs/scope/minimal_runtime.yaml",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dry-run", "offline", "online"], default="dry-run")
    p.add_argument("--out-root", type=str, default="outputs/scope_ablation")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated ablation names (default: all)",
    )
    args = p.parse_args(argv)

    names = list(ABLATIONS)
    if args.only:
        names = [x.strip() for x in args.only.split(",") if x.strip()]

    results = {}
    for name in names:
        cfg = ABLATIONS[name]
        out_dir = Path(args.out_root) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "training" / "train_scope.py"),
            "--mode",
            args.mode,
            "--config",
            cfg,
            "--out-dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
        metrics_path = out_dir / (
            "dry_run_metrics.json"
            if args.mode == "dry-run"
            else ("offline_metrics.json" if args.mode == "offline" else "online_metrics.json")
        )
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results[name] = {
            "returncode": proc.returncode,
            "metrics": metrics,
            "stderr_tail": (proc.stderr or "")[-500:],
        }

    summary_path = Path(args.out_root) / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(r["returncode"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
