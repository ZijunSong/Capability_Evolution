#!/usr/bin/env python3
"""Aggregate followup Phase B → PHASE_B_GATE.json + comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round10_followup"
PHASE_B = OUT / "phase_b"
MAIN = [
    "r10_main_noweight_seed42",
    "r10_main_noweight_seed43",
    "r10_main_noweight_seed44",
]
ALL = MAIN + [
    "r10_p0_exact_repro_seed42",
    "r10_natural_prior_noweight_seed42",
    "r10_balanced50_noweight_seed42",
    "r10_stage1_state_only_seed42",
    "r10_threshold_only_p0_seed42",
]


def _ok(v, floor: float) -> bool:
    return v is not None and float(v) >= floor


def load_variant(name: str) -> dict:
    path = PHASE_B / name / "TRAIN_AND_EVAL_REPORT.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # threshold-only may only have THRESHOLD report
    alt = PHASE_B / name / "THRESHOLD_ONLY_REPORT.json"
    if alt.exists():
        return {"variant": name, "threshold_only": json.loads(alt.read_text(encoding="utf-8"))}
    return {"variant": name}


def extract(rep: dict) -> dict:
    off = (rep.get("offline_valid") or {}).get("canonical_metrics") or {}
    hold = (rep.get("holdout") or {}).get("canonical_metrics") or {}
    # threshold-only / legacy hf_metrics fallback
    if not off:
        off = (rep.get("offline_valid") or {}).get("hf_metrics") or {}
    if not hold:
        hold = (rep.get("holdout") or {}).get("hf_metrics") or {}
    thr = rep.get("threshold_only")
    if not off and isinstance(thr, dict):
        off = thr.get("offline_valid") or thr.get("offline") or {}
        hold = thr.get("base_live") or thr.get("holdout") or {}
    parity_off = (rep.get("offline_valid") or {}).get("canonical_parity") or {}
    parity_hold = (rep.get("holdout") or {}).get("canonical_parity") or {}
    if rep.get("threshold_only") is True or isinstance(thr, (dict, bool)):
        # control arm — not held to canonical parity=1.0 in extract; gate only checks MAIN
        parity_off = parity_off or {"operation_agreement": None}
        parity_hold = parity_hold or {"operation_agreement": None}
    return {
        "variant": rep.get("variant"),
        "offline_operation_accuracy": off.get("operation_accuracy"),
        "offline_bal_acc": off.get("operation_balanced_accuracy"),
        "offline_ContinueRecall": off.get("ContinueRecall"),
        "offline_RollbackRecall": off.get("RollbackRecall"),
        "offline_pred_CONTINUE_prior": (off.get("prediction_prior") or {}).get("CONTINUE"),
        "offline_pred_ROLLBACK_prior": (off.get("prediction_prior") or {}).get("ROLLBACK_TO"),
        "holdout_bal_acc": hold.get("operation_balanced_accuracy"),
        "holdout_ContinueRecall": hold.get("ContinueRecall"),
        "holdout_RollbackRecall": hold.get("RollbackRecall"),
        "holdout_pred_CONTINUE_prior": (hold.get("prediction_prior") or {}).get("CONTINUE"),
        "holdout_pred_ROLLBACK_prior": (hold.get("prediction_prior") or {}).get("ROLLBACK_TO"),
        "checkpoint_top1": off.get("checkpoint_top1") if off.get("checkpoint_top1") is not None else hold.get("checkpoint_top1"),
        "checkpoint_mrr": off.get("checkpoint_mrr") if off.get("checkpoint_mrr") is not None else hold.get("checkpoint_mrr"),
        "gold_candidate_coverage": off.get("gold_candidate_coverage")
        if off.get("gold_candidate_coverage") is not None
        else hold.get("gold_candidate_coverage"),
        "canonical_parity_offline": parity_off.get("operation_agreement"),
        "canonical_parity_holdout": parity_hold.get("operation_agreement"),
        "fallback_offline": parity_off.get("fallback"),
        "fallback_holdout": parity_hold.get("fallback"),
    }


def main() -> None:
    rows = []
    for name in ALL:
        if not (PHASE_B / name).exists():
            continue
        rep = load_variant(name)
        rep.setdefault("variant", name)
        rows.append(extract(rep))

    main_rows = [r for r in rows if r["variant"] in MAIN]
    checks = []
    for r in main_rows:
        ok = (
            _ok(r["offline_bal_acc"], 0.80)
            and _ok(r["offline_ContinueRecall"], 0.70)
            and _ok(r["offline_RollbackRecall"], 0.70)
            and _ok(r["holdout_bal_acc"], 0.70)
            and _ok(r["holdout_ContinueRecall"], 0.70)
            and _ok(r["holdout_RollbackRecall"], 0.70)
            and _ok(r["checkpoint_top1"], 0.70)
            and _ok(r["checkpoint_mrr"], 0.85)
            and _ok(r["gold_candidate_coverage"], 0.99)
            and abs(float(r["canonical_parity_offline"] or 0) - 1.0) < 1e-12
            and abs(float(r["canonical_parity_holdout"] or 0) - 1.0) < 1e-12
        )
        checks.append({**r, "pass": ok})

    bals = [float(r["holdout_bal_acc"]) for r in main_rows if r["holdout_bal_acc"] is not None]
    span = (max(bals) - min(bals)) if bals else None
    gate_pass = bool(checks) and all(c["pass"] for c in checks) and (span is not None and span <= 0.05)

    # Comparative answers
    by_name = {r["variant"]: r for r in rows}
    p0 = by_name.get("r10_p0_exact_repro_seed42") or {}
    natural = by_name.get("r10_natural_prior_noweight_seed42") or {}
    bal50 = by_name.get("r10_balanced50_noweight_seed42") or {}
    stage1 = by_name.get("r10_stage1_state_only_seed42") or {}
    main42 = by_name.get("r10_main_noweight_seed42") or {}

    def _f(r, k):
        v = r.get(k)
        return None if v is None else float(v)

    answers = {
        "main_noweight_vs_p0_holdout_ContinueRecall": {
            "main42": _f(main42, "holdout_ContinueRecall"),
            "p0": _f(p0, "holdout_ContinueRecall"),
            "main_better": (
                _f(main42, "holdout_ContinueRecall") is not None
                and _f(p0, "holdout_ContinueRecall") is not None
                and _f(main42, "holdout_ContinueRecall") > _f(p0, "holdout_ContinueRecall")
            ),
        },
        "natural_vs_balanced50_holdout_ContinueRecall": {
            "natural": _f(natural, "holdout_ContinueRecall"),
            "balanced50": _f(bal50, "holdout_ContinueRecall"),
            "better": (
                "natural"
                if (_f(natural, "holdout_ContinueRecall") or -1)
                >= (_f(bal50, "holdout_ContinueRecall") or -1)
                else "balanced50"
            ),
        },
        "stage1_state_only_holdout_ContinueRecall": _f(stage1, "holdout_ContinueRecall"),
        "main_seed_span_holdout_bal_acc": span,
    }

    gate = {
        "pass": gate_pass,
        "STOP_BEFORE_CLOSED_LOOP": not gate_pass,
        "seed_span_operation_bal_acc_holdout": span,
        "main_seed_checks": checks,
        "all_variants": rows,
        "answers": answers,
        "canonical_inference_parity_required": 1.0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PHASE_B_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PHASE_B_COMPARISON",
        "",
        f"**Frozen Live Gate pass = {gate_pass}**",
        f"seed_span(holdout bal_acc) = {span}",
        "",
        "| variant | off_bal | off_CR | off_RR | live_bal | live_CR | live_RR | ck_top1 | ck_mrr | cov | parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(x):
            return "—" if x is None else f"{float(x):.4f}"
        lines.append(
            f"| {r['variant']} | {fmt(r['offline_bal_acc'])} | {fmt(r['offline_ContinueRecall'])} "
            f"| {fmt(r['offline_RollbackRecall'])} | {fmt(r['holdout_bal_acc'])} "
            f"| {fmt(r['holdout_ContinueRecall'])} | {fmt(r['holdout_RollbackRecall'])} "
            f"| {fmt(r['checkpoint_top1'])} | {fmt(r['checkpoint_mrr'])} "
            f"| {fmt(r['gold_candidate_coverage'])} | {fmt(r['canonical_parity_holdout'])} |"
        )
    lines += [
        "",
        "## Answers",
        "",
        f"- main_noweight vs P0 (live ContinueRecall): `{json.dumps(answers['main_noweight_vs_p0_holdout_ContinueRecall'])}`",
        f"- natural vs balanced50: `{json.dumps(answers['natural_vs_balanced50_holdout_ContinueRecall'])}`",
        f"- stage1_state_only live ContinueRecall: `{answers['stage1_state_only_holdout_ContinueRecall']}`",
        "",
        f"**STOP_BEFORE_CLOSED_LOOP = {not gate_pass}**",
        "",
    ]
    (OUT / "PHASE_B_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": gate_pass, "seed_span": span, "n_variants": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
