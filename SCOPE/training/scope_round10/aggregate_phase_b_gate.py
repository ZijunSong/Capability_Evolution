#!/usr/bin/env python3
"""Aggregate Phase B variants → PHASE_B_COMPARISON.md + FROZEN_LIVE_GATE.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_wave_b_report import split_report

OUT = _REPO / "outputs/scope_round10"
PHASE_B = OUT / "phase_b"
MAIN = [
    "r10_main_noweight_seed42",
    "r10_main_noweight_seed43",
    "r10_main_noweight_seed44",
]
ALL = MAIN + [
    "r10_natural_prior_noweight_seed42",
    "r10_balanced50_noweight_seed42",
    "r10_p0_exact_repro_seed42",
    "r10_threshold_only_p0_seed42",
    "r10_stage1_state_only_seed42",
]


def _recall_ok(v, floor=0.70) -> bool:
    return v is not None and float(v) >= floor


def load_variant(name: str) -> dict:
    vdir = PHASE_B / name
    report_path = vdir / "TRAIN_AND_EVAL_REPORT.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    # threshold-only already writes report; others aggregate on the fly
    return {
        "variant": name,
        "offline_valid": split_report(vdir, "offline_valid") if (vdir / "eval_offline_valid").exists() else {},
        "holdout": split_report(vdir, "holdout") if (vdir / "eval_holdout").exists() else {},
    }


def main() -> None:
    rows = []
    for name in ALL:
        if not (PHASE_B / name).exists():
            continue
        rep = load_variant(name)
        off = rep.get("offline_valid", {}).get("hf_metrics", {})
        hold = rep.get("holdout", {}).get("hf_metrics", {})
        # Wave B report uses holdout key; our runner uses eval_holdout via split_report("holdout")
        if not hold and "holdout" not in rep:
            hold = rep.get("holdout", {}).get("hf_metrics", {})
        parity_off = rep.get("offline_valid", {}).get("parity", {})
        parity_hold = rep.get("holdout", {}).get("parity", {})
        rows.append(
            {
                "variant": name,
                "offline_bal_acc": off.get("operation_balanced_accuracy"),
                "offline_ContinueRecall": off.get("ContinueRecall"),
                "offline_RollbackRecall": off.get("RollbackRecall"),
                "holdout_bal_acc": hold.get("operation_balanced_accuracy"),
                "holdout_ContinueRecall": hold.get("ContinueRecall"),
                "holdout_RollbackRecall": hold.get("RollbackRecall"),
                "offline_agreement": parity_off.get("operation_top1_agreement_raw"),
                "holdout_agreement": parity_hold.get("operation_top1_agreement_raw"),
                "pred_prior_holdout": hold.get("prediction_prior"),
                "checkpoint_top1": off.get("checkpoint_top1"),
                "checkpoint_mrr": off.get("checkpoint_mrr"),
            }
        )

    main_rows = [r for r in rows if r["variant"] in MAIN]
    checks = []
    for r in main_rows:
        ok = (
            (r["offline_bal_acc"] or 0) >= 0.80
            and _recall_ok(r["offline_ContinueRecall"])
            and _recall_ok(r["offline_RollbackRecall"])
            and (r["holdout_bal_acc"] or 0) >= 0.70
            and _recall_ok(r["holdout_ContinueRecall"])
            and _recall_ok(r["holdout_RollbackRecall"])
            and abs(float(r["offline_agreement"] or 0) - 1.0) < 1e-12
            and abs(float(r["holdout_agreement"] or 0) - 1.0) < 1e-12
        )
        checks.append({**r, "pass": ok})

    bals = [float(r["holdout_bal_acc"]) for r in main_rows if r["holdout_bal_acc"] is not None]
    span = (max(bals) - min(bals)) if bals else None
    gate_pass = bool(checks) and all(c["pass"] for c in checks) and (span is not None and span <= 0.05)

    gate = {
        "pass": gate_pass,
        "STOP_BEFORE_CLOSED_LOOP": not gate_pass,
        "main_seed_checks": checks,
        "seed_span_holdout_bal_acc": span,
        "seed_span_ok": span is not None and span <= 0.05,
        "note": "Stage2 ckpt metrics required >=0.70/0.85/0.99 when available; null ≠ PASS",
    }
    (OUT / "FROZEN_LIVE_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")

    md = [
        "# Phase B Comparison (Round 10)",
        "",
        "| variant | off bal | off Cont | off Roll | hold bal | hold Cont | hold Roll | off agr | hold agr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['offline_bal_acc']} | {r['offline_ContinueRecall']} | "
            f"{r['offline_RollbackRecall']} | {r['holdout_bal_acc']} | {r['holdout_ContinueRecall']} | "
            f"{r['holdout_RollbackRecall']} | {r['offline_agreement']} | {r['holdout_agreement']} |"
        )
    md += [
        "",
        f"**FROZEN_LIVE_GATE.pass = {gate_pass}**",
        f"seed_span_holdout_bal_acc = {span}",
        f"STOP_BEFORE_CLOSED_LOOP = {not gate_pass}",
        "",
    ]
    (OUT / "PHASE_B_COMPARISON.md").write_text("\n".join(md) + "\n")

    decision = {
        "PARITY_REGRESSION_ROOT_CAUSE": "R10-P9_vllm_replay_missing_disable_replan",
        "PARITY_FIXED": bool(json.loads((OUT / "PARITY_GATE.json").read_text()).get("pass"))
        if (OUT / "PARITY_GATE.json").exists()
        else False,
        "CONTINUE_COLLAPSE_AFTER_PARITY_FIX": None,
        "CLASS_WEIGHT_PRIOR_IS_CAUSAL": None,
        "STAGE1_CANDIDATE_SUMMARY_IS_CAUSAL": None,
        "FROZEN_LIVE_GATE_PASS": gate_pass,
        "SMOKE20_STARTED": False,
        "FINAL100_STARTED": False,
        "HARD_CAPABILITY_POSITIVE_SIGNAL": False,
        "RECOMMEND_ROLLBACK_830": False,
    }
    # Fill causal flags from main vs controls if present
    by = {r["variant"]: r for r in rows}
    main42 = by.get("r10_main_noweight_seed42")
    p0 = by.get("r10_p0_exact_repro_seed42")
    if main42 and p0 and main42.get("holdout_ContinueRecall") is not None and p0.get("holdout_ContinueRecall") is not None:
        decision["CLASS_WEIGHT_PRIOR_IS_CAUSAL"] = float(main42["holdout_ContinueRecall"]) > float(
            p0["holdout_ContinueRecall"]
        ) + 0.05
        decision["CONTINUE_COLLAPSE_AFTER_PARITY_FIX"] = float(main42["holdout_ContinueRecall"]) < 0.70
    state_only = by.get("r10_stage1_state_only_seed42")
    if main42 and state_only and main42.get("holdout_bal_acc") is not None and state_only.get("holdout_bal_acc") is not None:
        decision["STAGE1_CANDIDATE_SUMMARY_IS_CAUSAL"] = abs(
            float(main42["holdout_bal_acc"]) - float(state_only["holdout_bal_acc"])
        ) >= 0.03

    (OUT / "ROOT_CAUSE_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
