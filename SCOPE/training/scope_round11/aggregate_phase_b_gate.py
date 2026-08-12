#!/usr/bin/env python3
"""Aggregate Round11 Phase B and write FROZEN_LIVE_GATE.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round11"
MAIN = [
    "factorized_main_seed42",
    "factorized_main_seed43",
    "factorized_main_seed44",
]
ALL = MAIN + [
    "factorized_state_only_seed42",
    "factorized_full_stage1_seed42",
    "factorized_compact_signal_seed42",
    "factorized_ckpt_listwise_seed42",
    "factorized_ckpt_pairwise_seed42",
]


def _get(report: dict, split: str, key: str):
    block = report.get(split) or {}
    learned = block.get("learned_stage1_learned_stage2") or {}
    return learned.get(key)


def extract(variant: str) -> dict:
    path = OUT / "phase_b" / variant / "TRAIN_AND_EVAL_REPORT.json"
    if not path.exists():
        return {"variant": variant, "missing": True}
    report = json.loads(path.read_text(encoding="utf-8"))
    hold = report.get("holdout", {}).get("learned_stage1_learned_stage2") or {}
    off = report.get("offline_valid", {}).get("learned_stage1_learned_stage2") or {}
    return {
        "variant": variant,
        "missing": False,
        "stage1_view": (report.get("train") or {}).get("stage1_view"),
        "offline": {
            "balanced_accuracy": off.get("balanced_accuracy"),
            "ContinueRecall": off.get("ContinueRecall"),
            "RollbackRecall": off.get("RollbackRecall"),
            "checkpoint_top1": off.get("checkpoint_top1"),
            "checkpoint_mrr": off.get("checkpoint_mrr"),
            "gold_candidate_coverage": off.get("gold_candidate_coverage"),
            "canonical_parity": (off.get("canonical_parity") or {}).get("operation_agreement"),
        },
        "base_live": {
            "balanced_accuracy": hold.get("balanced_accuracy"),
            "ContinueRecall": hold.get("ContinueRecall"),
            "RollbackRecall": hold.get("RollbackRecall"),
            "prediction_prior": hold.get("prediction_prior"),
            "checkpoint_top1": hold.get("checkpoint_top1"),
            "checkpoint_mrr": hold.get("checkpoint_mrr"),
            "gold_candidate_coverage": hold.get("gold_candidate_coverage"),
            "invalid_checkpoint_rate": hold.get("invalid_checkpoint_rate"),
            "canonical_parity": (hold.get("canonical_parity") or {}).get("operation_agreement"),
        },
        "oracles_holdout": {
            "learned_s1_learned_s2": report.get("holdout", {}).get("learned_stage1_learned_stage2"),
            "oracle_s1_learned_s2": report.get("holdout", {}).get("oracle_stage1_learned_stage2"),
            "learned_s1_oracle_s2": report.get("holdout", {}).get("learned_stage1_oracle_stage2"),
            "oracle_s1_oracle_s2": report.get("holdout", {}).get("oracle_stage1_oracle_stage2"),
        },
    }


def seed_pass(row: dict) -> tuple[bool, list[str]]:
    fails = []
    off = row.get("offline") or {}
    live = row.get("base_live") or {}
    checks = [
        ("offline_bal>=0.80", float(off.get("balanced_accuracy") or 0) >= 0.80),
        ("offline_CR>=0.70", float(off.get("ContinueRecall") or 0) >= 0.70),
        ("offline_RR>=0.70", float(off.get("RollbackRecall") or 0) >= 0.70),
        ("live_bal>=0.70", float(live.get("balanced_accuracy") or 0) >= 0.70),
        ("live_CR>=0.70", float(live.get("ContinueRecall") or 0) >= 0.70),
        ("live_RR>=0.70", float(live.get("RollbackRecall") or 0) >= 0.70),
        ("ck_top1>=0.70", float(live.get("checkpoint_top1") or 0) >= 0.70),
        ("ck_mrr>=0.85", float(live.get("checkpoint_mrr") or 0) >= 0.85),
        ("cov>=0.99", float(live.get("gold_candidate_coverage") or 0) >= 0.99),
        ("parity==1.0", float(live.get("canonical_parity") or 0) >= 1.0 - 1e-9),
    ]
    for name, ok in checks:
        if not ok:
            fails.append(name)
    return len(fails) == 0, fails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    rows = [extract(v) for v in ALL]
    main_rows = [r for r in rows if r["variant"] in MAIN and not r.get("missing")]
    bals = [float((r.get("base_live") or {}).get("balanced_accuracy") or 0) for r in main_rows]
    span = (max(bals) - min(bals)) if bals else None
    per_seed = {}
    all_pass = True
    for r in main_rows:
        ok, fails = seed_pass(r)
        per_seed[r["variant"]] = {"pass": ok, "fails": fails, **r}
        all_pass = all_pass and ok
    span_ok = span is not None and span <= 0.05
    gate_pass = all_pass and span_ok and len(main_rows) == 3
    gate = {
        "pass": gate_pass,
        "STOP_AFTER_PHASE_B": (not gate_pass),
        "seed_span_base_live_balanced_accuracy": span,
        "seed_span_ok": span_ok,
        "main_seeds": per_seed,
        "all_variants": rows,
    }
    (OUT / "FROZEN_LIVE_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    # Comparison markdown
    lines = [
        "# PHASE_B_COMPARISON",
        "",
        f"FROZEN_LIVE_GATE.pass = **{gate_pass}**",
        f"STOP_AFTER_PHASE_B = **{not gate_pass}**",
        f"seed span(live bal) = {span}",
        "",
        "| variant | live_bal | live_CR | live_RR | off_bal | ck_top1 | ck_mrr | parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['variant']} | missing | | | | | | |")
            continue
        live = r["base_live"]
        off = r["offline"]
        lines.append(
            f"| {r['variant']} | {live.get('balanced_accuracy')} | {live.get('ContinueRecall')} | "
            f"{live.get('RollbackRecall')} | {off.get('balanced_accuracy')} | "
            f"{live.get('checkpoint_top1')} | {live.get('checkpoint_mrr')} | {live.get('canonical_parity')} |"
        )
    (OUT / "PHASE_B_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": gate_pass, "span": span, "n_main": len(main_rows)}, indent=2))


if __name__ == "__main__":
    main()
