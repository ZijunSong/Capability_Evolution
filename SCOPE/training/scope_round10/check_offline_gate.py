#!/usr/bin/env python3
"""Round 10 offline gate (Route B binary rollback)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import OUT, write_json
from training.scope_round9.aggregate_wave_b_report import split_report

TRAIN_OUT = OUT / "training"
MAIN = [
    "rollback_live_aligned_seed42",
    "rollback_live_aligned_seed43",
    "rollback_live_aligned_seed44",
]


def variant_report(vdir: Path, variant: str) -> dict:
    train_meta = {}
    tr = vdir / "train_only_report.json"
    if tr.exists():
        train_meta = json.loads(tr.read_text(encoding="utf-8"))
    offline = split_report(vdir, "offline_valid") if (vdir / "eval_offline_valid").exists() else {}
    live_valid = split_report(vdir, "live_valid") if (vdir / "eval_live_valid").exists() else {}
    live_test = split_report(vdir, "live_test") if (vdir / "eval_live_test").exists() else {}
    return {
        "variant": variant,
        "train_meta": train_meta,
        "offline_valid": offline,
        "live_valid": live_valid,
        "live_test": live_test,
    }


def check_variant(r: dict) -> dict:
    off = r.get("offline_valid", {}).get("hf_metrics", {})
    lt = r.get("live_test", {}).get("hf_metrics", {})
    parity = (
        r.get("offline_valid", {}).get("parity_pass", False)
        and r.get("live_test", {}).get("parity_pass", False)
    )
    row = {
        "variant": r["variant"],
        "offline_bal_acc": off.get("operation_balanced_accuracy", 0),
        "offline_ContinueRecall": off.get("ContinueRecall", 0),
        "offline_RollbackRecall": off.get("RollbackRecall", 0),
        "live_test_bal_acc": lt.get("operation_balanced_accuracy", 0),
        "live_test_ContinueRecall": lt.get("ContinueRecall", 0),
        "live_test_RollbackRecall": lt.get("RollbackRecall", 0),
        "parity_ok": parity,
    }
    row["pass"] = (
        row["offline_bal_acc"] >= 0.75
        and row["offline_ContinueRecall"] >= 0.70
        and row["offline_RollbackRecall"] >= 0.80
        and row["live_test_bal_acc"] >= 0.75
        and row["live_test_ContinueRecall"] >= 0.70
        and row["live_test_RollbackRecall"] >= 0.70
        and parity
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "ROUND10_OFFLINE_GATE.json")
    args = p.parse_args()

    reports = []
    all_checks = []
    for variant in MAIN:
        vdir = TRAIN_OUT / variant
        if not vdir.exists():
            continue
        r = variant_report(vdir, variant)
        reports.append(r)
        all_checks.append(check_variant(r))

    bals = [c["live_test_bal_acc"] for c in all_checks]
    span = max(bals) - min(bals) if bals else 999
    gate_pass = bool(all_checks) and all(c["pass"] for c in all_checks) and span <= 0.05

    result = {
        "route": "B",
        "main_seed_checks": all_checks,
        "seed_span_balanced_accuracy": span,
        "seed_span_ok": span <= 0.05,
        "offline_gate_pass": gate_pass,
        "variants": reports,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
