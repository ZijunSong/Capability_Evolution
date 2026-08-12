#!/usr/bin/env python3
"""Write Round11 ROOT_CAUSE_DECISION / ROUND11_REPORT / SHA256SUMS."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round11"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def main() -> None:
    phase_a = _load(OUT / "phase_a_state_factorization/PHASE_A_DECISION.json") or {}
    gate = _load(OUT / "FROZEN_LIVE_GATE.json") or {}
    per_view = _load(OUT / "phase_a_state_factorization/per_view_metrics.json") or {}
    selected = phase_a.get("selected_stage1_view")
    a0 = (per_view.get("A0") or {}).get("base_live") or {}
    a1 = (per_view.get("A1") or {}).get("base_live") or {}
    sel = (per_view.get(selected) or {}).get("base_live") or {}

    # Contamination signal: A0 vs A1 / A4 tradeoff
    contaminates = False
    if a0 and a1 and not a0.get("missing") and not a1.get("missing"):
        # Candidate semantics push toward ROLLBACK if A0 RR >> A1 RR and A0 CR << A1 CR
        contaminates = (
            float(a0.get("RollbackRecall") or 0) - float(a1.get("RollbackRecall") or 0) > 0.1
            and float(a1.get("ContinueRecall") or 0) - float(a0.get("ContinueRecall") or 0) > 0.1
        )

    main42 = (gate.get("main_seeds") or {}).get("factorized_main_seed42") or {}
    live = main42.get("base_live") or {}
    oracles = main42.get("oracles_holdout") or {}
    op_bot = ck_bot = None
    if oracles:
        l1l2 = oracles.get("learned_s1_learned_s2") or {}
        o1l2 = oracles.get("oracle_s1_learned_s2") or {}
        l1o2 = oracles.get("learned_s1_oracle_s2") or {}
        # If oracle Stage1 lifts more than oracle Stage2, operation dominates.
        if l1l2 and o1l2 and l1o2:
            d_op = float(o1l2.get("balanced_accuracy") or 0) - float(l1l2.get("balanced_accuracy") or 0)
            d_ck = float(l1o2.get("balanced_accuracy") or 0) - float(l1l2.get("balanced_accuracy") or 0)
            op_bot = d_op >= d_ck
            ck_bot = d_ck > d_op

    root = {
        "selected_stage1_view": selected,
        "checkpoint_semantic_contaminates_stage1": contaminates,
        "frozen_live_gate_pass": bool(gate.get("pass")),
        "STOP_AFTER_PHASE_B": bool(gate.get("STOP_AFTER_PHASE_B", True)),
        "dominant_bottleneck": (
            "operation" if op_bot else ("checkpoint" if ck_bot else "unknown_or_both")
        ),
        "rollback_hard_capability_established": False,  # requires Phase D
        "answers": {
            "1_checkpoint_semantic_contamination": contaminates,
            "2_best_minimal_stage1_view": selected,
            "3_factorization_improves_continue_recall": None,
            "4_stage2_ranker_reaches_0.70_top1": float(live.get("checkpoint_top1") or 0) >= 0.70,
            "5_dominant_bottleneck": "operation" if op_bot else ("checkpoint" if ck_bot else "unknown"),
            "6_rollback_hard_on_100q": False,
        },
        "selected_view_base_live": sel,
        "a0_base_live": a0,
        "a1_base_live": a1,
    }
    # Fill Q3 from main vs r10 if available
    try:
        r10 = json.loads(
            (_REPO / "outputs/scope_round10_followup/phase_b/r10_main_noweight_seed42/TRAIN_AND_EVAL_REPORT.json").read_text()
        )
        r10_cr = (
            ((r10.get("holdout") or {}).get("canonical_metrics") or {}).get("ContinueRecall")
        )
        root["answers"]["3_factorization_improves_continue_recall"] = (
            float(live.get("ContinueRecall") or 0) > float(r10_cr or 0)
            if r10_cr is not None
            else None
        )
        root["r10_main_live_ContinueRecall"] = r10_cr
        root["r11_main_live_ContinueRecall"] = live.get("ContinueRecall")
    except Exception:
        pass

    (OUT / "ROOT_CAUSE_DECISION.json").write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")

    report = f"""# ROUND11_REPORT

## Summary

- Selected Stage1 view: `{selected}`
- FROZEN_LIVE_GATE.pass: `{gate.get('pass')}`
- STOP_AFTER_PHASE_B: `{gate.get('STOP_AFTER_PHASE_B')}`
- Checkpoint semantic contamination (A0 vs A1): `{contaminates}`
- Dominant bottleneck: `{root['dominant_bottleneck']}`
- Rollback hard capability established: `false` (requires closed-loop 100q gate)

## Phase A

See `FEATURE_FACTOR_REPORT.md` and `phase_a_state_factorization/`.

## Phase B

See `PHASE_B_COMPARISON.md` and `FROZEN_LIVE_GATE.json`.

## Required answers

1. Checkpoint semantic content contaminate Stage1? **{contaminates}**
2. Best minimal Stage1 feature group? **{selected}**
3. Factorizing improve frozen-live ContinueRecall? **{root['answers']['3_factorization_improves_continue_recall']}**
4. Stage2 ranker top1 >= 0.70? **{root['answers']['4_stage2_ranker_reaches_0.70_top1']}**
5. Dominant bottleneck? **{root['answers']['5_dominant_bottleneck']}**
6. Rollback hard on 100q? **false** (not run / gate failed)
"""
    (OUT / "ROUND11_REPORT.md").write_text(report, encoding="utf-8")

    # SHA256 of key artifacts
    paths = []
    for pat in [
        "ENVIRONMENT_SNAPSHOT.txt",
        "RUN_MANIFEST.json",
        "FEATURE_FACTOR_REPORT.md",
        "PHASE_B_COMPARISON.md",
        "FROZEN_LIVE_GATE.json",
        "ROOT_CAUSE_DECISION.json",
        "ROUND11_REPORT.md",
        "phase_a_state_factorization/per_view_metrics.json",
        "phase_a_state_factorization/PHASE_A_DECISION.json",
    ]:
        p = OUT / pat
        if p.exists():
            paths.append(p)
    lines = []
    for p in paths:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT / "ROUND11_REPORT.md"), "gate": gate.get("pass")}, indent=2))


if __name__ == "__main__":
    main()
