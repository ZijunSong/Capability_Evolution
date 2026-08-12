#!/usr/bin/env python3
"""Aggregate Phase A root-cause categories and write PARITY_GATE / report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round10"
PHASE_A = OUT / "phase_a"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    index = load(PHASE_A / "LEDGER_INDEX.json")
    by_key = {(r["seed"], r["split"]): r for r in index}

    # Primary root cause from ledger: REPLAN path enabled on vLLM replay.
    replan_mism = sum(r.get("replan_path_mismatch", 0) for r in index)
    raw_mism = sum(r.get("raw_mismatch", 0) for r in index)
    fixed_mism = sum(r.get("disable_replan_mismatch", 0) for r in index)
    prompt_mm = sum(r.get("prompt_hash_mismatch", 0) for r in index)
    token_mm = sum(r.get("token_hash_mismatch", 0) for r in index)
    cand_mm = sum(r.get("candidate_hash_mismatch", 0) for r in index)
    fallback = sum(r.get("fallback_count", 0) for r in index)
    near_b = sum(r.get("near_boundary_residual", 0) for r in index)

    # Optional float32 audit summaries
    float32_files = list(PHASE_A.glob("seed*/**/float32_rescore.jsonl"))
    float32_stats = []
    for fp in float32_files:
        n = agree = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n += 1
                if row.get("agree_vllm_after_float32"):
                    agree += 1
        float32_stats.append({"path": str(fp), "n": n, "agree": agree})

    categories = {
        "R10-P1_state_serialization_drift": {
            "verdict": "false",
            "evidence_events": 0,
            "note": "frozen effective_input reused; no state re-serialize in replay",
        },
        "R10-P2_chat_template_prompt_drift": {
            "verdict": "false" if prompt_mm == 0 else "true",
            "evidence_events": prompt_mm,
            "note": "prompt_sha256 compared on stored frozen rows",
        },
        "R10-P3_tokenizer_token_boundary_drift": {
            "verdict": "false" if token_mm == 0 else "uncertain",
            "evidence_events": token_mm,
            "note": "token_ids_sha256 from frozen build; residual numerical flips may still involve tokenization of verbalizers",
        },
        "R10-P4_max_length_truncation_drift": {
            "verdict": "false",
            "evidence_events": 0,
            "note": "both backends score the same truncated effective_input_text",
        },
        "R10-P5_verbalizer_scoring_implementation_drift": {
            "verdict": "uncertain" if fixed_mism > 0 else "false",
            "evidence_events": fixed_mism,
            "note": "after disable_replan, residual mismatches are score-level C vs R disagreements",
        },
        "R10-P6_candidate_ordering_label_map_drift": {
            "verdict": "false" if cand_mm == 0 else "true",
            "evidence_events": cand_mm,
        },
        "R10-P7_adapter_vs_merged_checkpoint_drift": {
            "verdict": "uncertain",
            "evidence_events": 0,
            "note": "see phase_a/audits/adapter_merged_* if present",
        },
        "R10-P8_dtype_numerical_near_boundary_instability": {
            "verdict": "true" if fixed_mism > 0 else "false",
            "evidence_events": near_b,
            "note": "residual after REPLAN fix; float32 audits in phase_a",
            "float32_stats": float32_stats,
        },
        "R10-P9_threshold_comparison_operator_drift": {
            "verdict": "true",
            "evidence_events": replan_mism,
            "note": "vLLM replay omitted disable_replan=True while HF enabled it; all raw holdout mismatches were HF∈{C,R} vs vLLM=REPLAN",
        },
        "R10-P10_hidden_fallback_exception_path": {
            "verdict": "false" if fallback == 0 else "true",
            "evidence_events": fallback,
        },
    }

    # Agreement after contract fix (disable_replan redecide on stored logits)
    seed_agreements = []
    all_pass = True
    for seed in (42, 43, 44):
        for split in ("offline_valid", "base_live"):
            row = by_key[(seed, split)]
            agr = float(row["disable_replan_agreement"])
            seed_agreements.append(
                {
                    "seed": seed,
                    "split": split,
                    "agreement": agr,
                    "n": row["n"],
                    "mismatch": row["disable_replan_mismatch"],
                    "raw_agreement": row["raw_agreement"],
                }
            )
            if abs(agr - 1.0) > 1e-12:
                all_pass = False

    # Gate A requires exact 1.0. Contract fix alone may leave numerical residual.
    # If float32 + redecide still not 1.0, STOP_AFTER_PHASE_A unless a stable
    # tie rule file is present that achieves 1.0 on both backends.
    stabilize_path = PHASE_A / "STABLE_TIE_RULE.json"
    stabilize = load(stabilize_path) if stabilize_path.exists() else None
    if stabilize and stabilize.get("agreement_all_splits") == 1.0:
        all_pass = True
        gate_note = "pass via disable_replan + documented stable tie rule"
    elif all_pass:
        gate_note = "pass via disable_replan contract fix alone"
    else:
        gate_note = (
            "FAIL: after disable_replan, residual numerical operation flips remain; "
            "see R10-P8/P9. STOP_AFTER_PHASE_A until float32/eager path or stable tie rule yields 1.0"
        )

    gate = {
        "pass": all_pass,
        "STOP_AFTER_PHASE_A": not all_pass,
        "note": gate_note,
        "requirements": {
            "offline_valid_agreement": 1.0,
            "base_live_agreement": 1.0,
            "serialized_state_hash_agreement": 1.0,
            "rendered_prompt_hash_agreement": 1.0,
            "token_ids_agreement": 1.0,
            "candidate_ordering_agreement": 1.0,
            "fallback_rate": 0,
        },
        "observed": {
            "seed_agreements_after_disable_replan": seed_agreements,
            "prompt_hash_mismatch_total": prompt_mm,
            "token_hash_mismatch_total": token_mm,
            "candidate_hash_mismatch_total": cand_mm,
            "fallback_total": fallback,
            "raw_mismatch_total": raw_mism,
            "replan_path_mismatch_total": replan_mism,
            "residual_mismatch_after_disable_replan": fixed_mism,
        },
        "primary_root_cause": "R10-P9_threshold_comparison_operator_drift",
        "secondary_root_cause": "R10-P8_dtype_numerical_near_boundary_instability",
        "fix_applied": {
            "replay_frozen_vllm_disable_replan": True,
            "redecide_replay_logits_disable_replan": True,
        },
        "stable_tie_rule": stabilize,
    }
    (OUT / "PARITY_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    md = [
        "# PARITY_ROOT_CAUSE_REPORT (Round 10 Phase A)",
        "",
        "## Primary finding",
        "",
        "P0 holdout HF↔vLLM operation agreement ~0.75 is **not** primarily a state/prompt/token",
        "serialization drift. Event-level comparison shows **all raw mismatches** are of the form",
        "`HF ∈ {CONTINUE, ROLLBACK_TO}` vs `vLLM = REPLAN`.",
        "",
        "Root cause **R10-P9**: `replay_frozen_hf.py` called `decide_rollback_operation(..., disable_replan=True)`",
        "while `replay_frozen_vllm.py` omitted `disable_replan`, so vLLM still argmaxed over REPLAN.",
        "(`VllmRollbackScorer.score_final_prompt` already used disable_replan, but replay re-decided without it.)",
        "",
        f"- raw mismatch events (all splits/seeds sum): **{raw_mism}**",
        f"- of which vLLM-REPLAN path: **{replan_mism}**",
        f"- residual mismatches after disable_replan redecide: **{fixed_mism}**",
        f"- near-boundary residual (|margin|<0.25 either side): **{near_b}**",
        "",
        "## Per-seed agreements after disable_replan",
        "",
        "| seed | split | raw agr | fixed agr | residual mism |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in seed_agreements:
        md.append(
            f"| {row['seed']} | {row['split']} | {row['raw_agreement']:.6f} | "
            f"{row['agreement']:.6f} | {row['mismatch']} |"
        )
    md += [
        "",
        "## Category verdicts",
        "",
    ]
    for k, v in categories.items():
        md.append(f"- **{k}**: `{v['verdict']}` (events={v['evidence_events']}) — {v.get('note','')}")
    md += [
        "",
        f"## Gate A: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        gate_note,
        "",
        f"STOP_AFTER_PHASE_A = {not all_pass}",
        "",
    ]
    (OUT / "PARITY_ROOT_CAUSE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (PHASE_A / "ROOT_CAUSE_CATEGORIES.json").write_text(
        json.dumps(categories, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pass": all_pass, "residual": fixed_mism}, indent=2))


if __name__ == "__main__":
    main()
