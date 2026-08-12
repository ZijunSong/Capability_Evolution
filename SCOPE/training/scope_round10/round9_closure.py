#!/usr/bin/env python3
"""Barrier 0: Round 9 engineering closure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import OUT, R9_OUT, load_jsonl, write_json
from training.scope_round9.aggregate_oracle_factorization import eval_mode, MODES
from training.scope_round9.aggregate_phase3_gate import _checkpoint_metrics

WAVE_B = R9_OUT / "wave_b"
CLOSURE = OUT / "round9_closure"


def run_diagnosis_report() -> None:
    subprocess.run(
        [
            sys.executable,
            "training/scope_round9/write_round9_diagnosis_report.py",
            "--output",
            str(R9_OUT / "ROUND9_DIAGNOSIS_REPORT.md"),
        ],
        cwd=_REPO,
        check=True,
    )


def supplement_stage2_metrics() -> dict:
    results = {}
    for vdir in sorted(WAVE_B.iterdir()):
        if not vdir.is_dir():
            continue
        holdout_hf = vdir / "eval_holdout" / "hf_replay.jsonl"
        if not holdout_hf.exists():
            continue
        rows = load_jsonl(holdout_hf)
        events = [
            {
                "shadow_operation": r.get("gold_operation"),
                "student_operation": r.get("pred_operation"),
                "shadow_checkpoint_id": r.get("gold_checkpoint_global_id"),
                "predicted_checkpoint_id": r.get("pred_checkpoint_global_id"),
                "candidate_checkpoint_ids": [
                    c.get("checkpoint_id") for c in (r.get("candidate_list") or [])
                ],
            }
            for r in rows
        ]
        ck = _checkpoint_metrics(events)
        results[vdir.name] = {
            "checkpoint_candidate_coverage": ck["checkpoint_candidate_coverage"],
            "checkpoint_top1": ck["checkpoint_accuracy"],
            "checkpoint_mrr": ck["checkpoint_mrr"],
            "invalid_checkpoint_rate": ck.get("invalid_checkpoint_rate", 0.0),
            "n_events": len(events),
        }
    write_json(CLOSURE / "stage2_checkpoint_metrics.json", results)
    return results


def fix_root_cause_decision() -> dict:
    root_path = R9_OUT / "ROOT_CAUSE_DECISION.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    modes = root.get("modes", {})
    full = modes.get("learned_op + learned_ckpt", {})
    op_oracle = modes.get("learned_op + oracle_ckpt", {})
    ck_oracle = modes.get("oracle_op + learned_ckpt", {})

    bottlenecks = []
    if full.get("ContinueRecall", 1) < 0.5:
        bottlenecks.append("continue_collapse")
    if abs(full.get("RollbackRecall", 0) - op_oracle.get("RollbackRecall", 0)) > 0.1:
        bottlenecks.append("live_operation_distribution_shift")
    bottlenecks.append("replan_support_missing")

    secondary = []
    if ck_oracle.get("checkpoint_top1", 1) >= 0.85:
        secondary.append("holdout_hf_vllm_near_boundary_parity")

    diagnosis = root.setdefault("diagnosis", {})
    diagnosis["primary_bottlenecks"] = bottlenecks
    diagnosis["secondary_bottlenecks"] = secondary
    diagnosis["evidence"] = {
        "learned_continue_recall": full.get("ContinueRecall"),
        "learned_rollback_recall": full.get("RollbackRecall"),
        "oracle_ckpt_top1": ck_oracle.get("checkpoint_top1"),
        "operation_bal_acc": full.get("operation_balanced_accuracy"),
    }
    root_path.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")
    write_json(CLOSURE / "ROOT_CAUSE_DECISION_PATCH.json", diagnosis)
    return diagnosis


def export_parity_disagreements() -> dict:
    variants = ["rollback_correct_only", "rollback_soft_replan_only"]
    events_out = []
    summary = {}
    for variant in variants:
        hf = R9_OUT / "wave_a" / variant / "base_live" / "hf_replay.jsonl"
        vllm = R9_OUT / "wave_a" / variant / "base_live" / "vllm_replay.jsonl"
        if not hf.exists() or not vllm.exists():
            continue
        hf_rows = {r.get("event_id", i): r for i, r in enumerate(load_jsonl(hf))}
        vllm_rows = {r.get("event_id", i): r for i, r in enumerate(load_jsonl(vllm))}
        disagree = 0
        near_margin = 0
        for eid, hr in hf_rows.items():
            vr = vllm_rows.get(eid)
            if not vr:
                continue
            if hr.get("pred_operation") != vr.get("pred_operation"):
                disagree += 1
                margin = abs(
                    float(hr.get("margin_rollback", 0) or 0)
                    - float(hr.get("margin_continue", 0) or 0)
                )
                if margin < 0.01:
                    near_margin += 1
                events_out.append(
                    {
                        "variant": variant,
                        "event_id": eid,
                        "effective_prompt": hr.get("effective_input_text", "")[:500],
                        "hf_pred": hr.get("pred_operation"),
                        "vllm_pred": vr.get("pred_operation"),
                        "top1_margin": margin,
                        "prompt_sha256": hr.get("prompt_sha256"),
                        "token_ids_sha256": hr.get("token_ids_sha256"),
                    }
                )
        agr = 1 - disagree / max(len(hf_rows), 1)
        summary[variant] = {
            "hf_vllm_top1_agreement": agr,
            "disagreement_count": disagree,
            "near_zero_margin_disagreements": near_margin,
            "all_near_margin": disagree == 0 or near_margin == disagree,
        }

    # Wave B holdout parity scan
    for vdir in sorted(WAVE_B.iterdir()):
        if not vdir.is_dir():
            continue
        hf = vdir / "eval_holdout" / "hf_replay.jsonl"
        vllm = vdir / "eval_holdout" / "vllm_replay.jsonl"
        if not hf.exists() or not vllm.exists():
            continue
        hf_rows = load_jsonl(hf)
        vllm_rows = load_jsonl(vllm)
        agree = sum(
            1
            for h, v in zip(hf_rows, vllm_rows)
            if h.get("pred_operation") == v.get("pred_operation")
        )
        summary[vdir.name] = {
            "hf_vllm_top1_agreement": agree / max(len(hf_rows), 1),
            "n": len(hf_rows),
        }

    write_json(CLOSURE / "parity_disagreement_summary.json", summary)
    write_jsonl_path = CLOSURE / "parity_disagreement_events.jsonl"
    write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with write_jsonl_path.open("w", encoding="utf-8") as f:
        for ev in events_out:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()
    CLOSURE.mkdir(parents=True, exist_ok=True)
    run_diagnosis_report()
    stage2 = supplement_stage2_metrics()
    diagnosis = fix_root_cause_decision()
    parity = export_parity_disagreements()
    write_json(
        CLOSURE / "BARRIER0_DONE.json",
        {"stage2": stage2, "diagnosis": diagnosis, "parity": parity},
    )
    print("Barrier 0 complete")


if __name__ == "__main__":
    main()
