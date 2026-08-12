#!/usr/bin/env python3
"""Aggregate Wave C smoke/final gates and Round 9 final reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_phase3_gate import aggregate_run, evaluate_hard_gate, merge_shards

OUT = _REPO / "outputs/scope_round9"
MAIN_SEEDS = [
    "rollback_hier_o7_seed42",
    "rollback_hier_o7_seed43",
    "rollback_hier_o7_seed44",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke20", "final100"], required=True)
    args = p.parse_args()

    root = OUT / "wave_c" / args.mode
    variants: dict = {}
    for child in sorted(root.iterdir()) if root.exists() else []:
        if not child.is_dir():
            continue
        if args.mode == "smoke20":
            variants[child.name] = aggregate_run(child)
        else:
            agg = merge_shards(child)
            variants[child.name] = aggregate_run(agg)

    gate = evaluate_hard_gate(variants)
    report = {
        "mode": args.mode,
        "variants": variants,
        **gate,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
    }

    if args.mode == "smoke20":
        out_path = OUT / "H20_SMOKE_GATE.json"
        seeds = [variants.get(v, {}) for v in MAIN_SEEDS if v in variants]
        report["smoke_pass"] = all(
            m.get("ContinueRecall", 0) >= 0.1
            and m.get("RollbackRecall", 0) >= 0.30
            and m.get("FalseRollbackRate", 0) <= 0.05
            for m in seeds
        )
    else:
        out_path = OUT / "HARD_CAPABILITY_GATE_ROUND9.json"
        report["ROUND9_HARD_CAPABILITY_POSITIVE"] = gate.get("hard_capability_positive_signal", False)
        report["RECOMMEND_ROLLBACK_830"] = gate.get("hard_capability_positive_signal", False)

    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
