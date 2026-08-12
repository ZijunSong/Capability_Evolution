#!/usr/bin/env python3
"""Aggregate Wave B reports into OFFLINE_GATE_ROUND9.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_wave_b_report import offline_gate

OUT = _REPO / "outputs/scope_round9/wave_b"
MAIN = [
    "rollback_hier_o7_seed42",
    "rollback_hier_o7_seed43",
    "rollback_hier_o7_seed44",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    reports = []
    all_variants = {}
    for path in sorted(OUT.glob("*/TRAIN_AND_EVAL_REPORT.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        all_variants[data.get("variant", path.parent.name)] = data
        if path.parent.name in MAIN:
            reports.append(data)

    gate = offline_gate(reports)
    # Seed span check on offline bal_acc
    bal = [r.get("offline_valid", {}).get("hf_metrics", {}).get("operation_balanced_accuracy", 0) for r in reports]
    span = (max(bal) - min(bal)) if bal else 999
    gate["seed_span_operation_bal_acc"] = span
    gate["seed_span_ok"] = span <= 0.05
    gate["offline_gate_pass"] = bool(gate.get("offline_gate_pass")) and gate["seed_span_ok"]
    gate["variants"] = {k: {
        "offline_bal_acc": v.get("offline_valid", {}).get("hf_metrics", {}).get("operation_balanced_accuracy"),
        "holdout_bal_acc": v.get("holdout", {}).get("hf_metrics", {}).get("operation_balanced_accuracy"),
        "offline_ContinueRecall": v.get("offline_valid", {}).get("hf_metrics", {}).get("ContinueRecall"),
        "holdout_ContinueRecall": v.get("holdout", {}).get("hf_metrics", {}).get("ContinueRecall"),
        "parity_offline": v.get("offline_valid", {}).get("parity_pass"),
        "parity_holdout": v.get("holdout", {}).get("parity_pass"),
    } for k, v in all_variants.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if not gate["offline_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
