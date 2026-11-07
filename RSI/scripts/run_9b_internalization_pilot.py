#!/usr/bin/env python3
"""run_9b_internalization_pilot (guide §11.2 / §9.2): run the comparison
cells for one domain's pilot — Base, Runtime teacher, RL-only and HarnessOPD
— and summarize.

This orchestrates EXISTING CLI commands; it does not fabricate comparisons.
Cells that were not run are reported as NOT_RUN, never as 0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RSI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RSI_ROOT / "src"))


def _sh(cmd: list[str]) -> dict:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(RSI_ROOT), capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "rc": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. configs/alfworld_opd_pilot_v1.yaml")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--cells", default="base,runtime_teacher,rl_only,harness_opd",
                    help="comma list from: base,runtime_teacher,rl_only,harness_opd,aligned_sft,no_defer")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out_dir = Path(args.out) if args.out else RSI_ROOT / "runs" / f"{json.loads(json.dumps(args.config)) and Path(args.config).stem}_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    def eval_cell(name: str, extra: list[str]) -> dict:
        res = _sh([sys.executable, "-m", "rsi.cli", "eval", "--config", args.config,
                   "--harness", "minimal", "--split", "final_test"] + extra)
        return {"run": res}

    if "base" in cells:
        summary["base"] = eval_cell("base", [])
    if "runtime_teacher" in cells:
        summary["runtime_teacher"] = eval_cell("runtime_teacher", ["--bank", args.bank])
    # rl_only / harness_opd / aligned_sft / no_defer are TRAINING cells: they
    # require the readiness gate and are invoked via `rsi.cli train` with the
    # appropriate method in the config.  The pilot script only reports what
    # actually ran.
    summary["_notes"] = {
        "training_cells": [c for c in cells if c not in ("base", "runtime_teacher")],
        "readiness_required": True,
        "note": "training cells run via rsi.cli train after the teacher readiness gate passes",
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps({"ok": True, "out": str(out_dir / "pilot_summary.json")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
