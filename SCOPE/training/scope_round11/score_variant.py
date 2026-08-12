#!/usr/bin/env python3
"""Score Round11 factorized variant + oracle combinations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.canonical_rollback_scorer import decide_from_saved_logits
from training.scope_round9.aggregate_frozen_replay import load_jsonl, operation_metrics
from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy


def _recall(matrix: dict, op: str) -> float | None:
    support = sum(matrix.get(op, {}).values()) if isinstance(matrix.get(op), dict) else 0
    if support <= 0:
        return None
    return matrix[op][op] / support


def _checkpoint_metrics(rows: list[dict]) -> dict:
    eligible = 0
    covered = 0
    correct = 0
    mrr_sum = 0.0
    invalid = 0
    for r in rows:
        if r.get("gold_operation") != "ROLLBACK_TO":
            continue
        eligible += 1
        gold_ck = r.get("gold_checkpoint_global_id")
        candidates = [c.get("checkpoint_id") for c in (r.get("candidate_list") or [])]
        gold_in = bool(r.get("gold_in_candidates")) or (gold_ck in candidates)
        if gold_in:
            covered += 1
        # Prefer learned rank scores for MRR when present.
        rank_scores = r.get("checkpoint_rank_scores") or []
        ranked_ids = [x.get("checkpoint_id") for x in rank_scores] if rank_scores else candidates
        if r.get("pred_operation") == "ROLLBACK_TO":
            pred_ck = r.get("pred_checkpoint_global_id")
            if pred_ck not in set(candidates):
                invalid += 1
            if pred_ck == gold_ck:
                correct += 1
        if ranked_ids and gold_ck in ranked_ids:
            rank = ranked_ids.index(gold_ck) + 1
            mrr_sum += 1.0 / rank
        elif candidates and gold_ck in candidates:
            rank = candidates.index(gold_ck) + 1
            mrr_sum += 1.0 / rank
    return {
        "checkpoint_top1": correct / max(eligible, 1),
        "checkpoint_mrr": mrr_sum / max(eligible, 1),
        "gold_candidate_coverage": covered / max(eligible, 1),
        "invalid_checkpoint_rate": invalid / max(eligible, 1),
        "n_checkpoint_eval": eligible,
    }


def split_metrics(rows: list[dict]) -> dict:
    metrics = operation_metrics(rows)
    matrix = metrics["confusion_matrix"]
    prior_counts = Counter(r.get("pred_operation") for r in rows)
    n = max(len(rows), 1)
    prior = {k: prior_counts.get(k, 0) / n for k in ("CONTINUE", "ROLLBACK_TO", "REPLAN")}
    bal = _balanced_accuracy(matrix)
    mismatch = 0
    fallback = 0
    replan = 0
    for r in rows:
        if r.get("vllm_logits"):
            b = decide_from_saved_logits(r, logits_key="vllm_logits", disable_replan=True)
            # Only check operation parity when not oracle-overridden
            if not r.get("operation_oracle") and b.pred_operation != r.get("pred_operation"):
                # If stage2-only override of ck, operation should still match
                if b.pred_operation != r.get("pred_operation"):
                    mismatch += 1
            if b.fallback_reason or r.get("fallback_reason"):
                fallback += 1
            if b.pred_operation == "REPLAN":
                replan += 1
    return {
        "n": len(rows),
        "balanced_accuracy": float(bal),
        "ContinueRecall": _recall(matrix, "CONTINUE"),
        "RollbackRecall": _recall(matrix, "ROLLBACK_TO"),
        "prediction_prior": prior,
        **_checkpoint_metrics(rows),
        "canonical_parity": {
            "operation_agreement": 1.0 - mismatch / n,
            "mismatch": mismatch,
            "fallback": fallback,
            "disable_replan_violations": replan,
            "pass": mismatch == 0 and replan == 0,
        },
        "confusion_matrix": matrix,
    }


def oracle_combo_metrics(rows: list[dict], *, op_oracle: bool, ck_oracle: bool) -> dict:
    combo = []
    for r in rows:
        pred_op = r.get("gold_operation") if op_oracle else r.get("pred_operation")
        pred_ck = r.get("gold_checkpoint_global_id") if ck_oracle else r.get("pred_checkpoint_global_id")
        pred_ck_l = r.get("gold_checkpoint_local_id") if ck_oracle else r.get("pred_checkpoint_local_id")
        combo.append(
            {
                **r,
                "pred_operation": pred_op,
                "pred_checkpoint_global_id": pred_ck if pred_op == "ROLLBACK_TO" else None,
                "pred_checkpoint_local_id": pred_ck_l if pred_op == "ROLLBACK_TO" else None,
                "operation_oracle": op_oracle,
            }
        )
    return split_metrics(combo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    train_meta = {}
    train_only = args.variant_dir / "train_only_report.json"
    if train_only.exists():
        train_meta = json.loads(train_only.read_text(encoding="utf-8"))

    report: dict = {"variant": args.variant, "train": train_meta, "scorer_backend": "vllm_canonical_factorized"}
    for split in ("offline_valid", "holdout"):
        path = args.variant_dir / f"eval_{split}" / "canonical_vllm_replay.jsonl"
        if not path.exists():
            report[split] = {}
            continue
        rows = load_jsonl(path)
        report[split] = {
            "learned_stage1_learned_stage2": split_metrics(rows),
            "oracle_stage1_learned_stage2": oracle_combo_metrics(rows, op_oracle=True, ck_oracle=False),
            "learned_stage1_oracle_stage2": oracle_combo_metrics(rows, op_oracle=False, ck_oracle=True),
            "oracle_stage1_oracle_stage2": oracle_combo_metrics(rows, op_oracle=True, ck_oracle=True),
            "n": len(rows),
            "source": str(path),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant": args.variant, "wrote": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
