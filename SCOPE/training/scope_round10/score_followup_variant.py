#!/usr/bin/env python3
"""Score a followup Phase B variant from canonical vLLM replay outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.canonical_rollback_scorer import decide_from_saved_logits
from training.scope_round9.aggregate_frozen_replay import load_jsonl, operation_metrics
from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy, _confusion_matrix


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
    for r in rows:
        if r.get("gold_operation") != "ROLLBACK_TO":
            continue
        gold_ck = r.get("gold_checkpoint_global_id")
        candidates = [c.get("checkpoint_id") for c in (r.get("candidate_list") or [])]
        gold_in = bool(r.get("gold_in_candidates"))
        if candidates and gold_ck not in candidates and not gold_in:
            # skip uncovered for top1 denom? still count coverage denom
            eligible += 1
            continue
        eligible += 1
        if gold_in or (candidates and gold_ck in candidates):
            covered += 1
        if r.get("pred_operation") != "ROLLBACK_TO":
            continue
        pred_ck = r.get("pred_checkpoint_global_id")
        if pred_ck == gold_ck:
            correct += 1
        if candidates and gold_ck in candidates:
            try:
                rank = candidates.index(gold_ck) + 1
                mrr_sum += 1.0 / rank
            except ValueError:
                pass
    return {
        "checkpoint_top1": correct / max(eligible, 1),
        "checkpoint_mrr": mrr_sum / max(eligible, 1),
        "gold_candidate_coverage": covered / max(eligible, 1),
        "n_checkpoint_eval": eligible,
    }


def split_metrics(variant_dir: Path, split: str) -> dict:
    path = variant_dir / f"eval_{split}" / "canonical_vllm_replay.jsonl"
    alt = variant_dir / f"eval_{split}" / "vllm_replay.jsonl"
    src = path if path.exists() else alt
    if not src.exists():
        return {}
    rows = load_jsonl(src)
    # self-consistency via canonical redecide
    redecided = []
    mismatch = 0
    fallback = 0
    replan = 0
    for r in rows:
        b = decide_from_saved_logits(r, logits_key="vllm_logits", disable_replan=True)
        if b.pred_operation != r.get("pred_operation"):
            mismatch += 1
        if b.fallback_reason or r.get("fallback_reason"):
            fallback += 1
        if b.pred_operation == "REPLAN":
            replan += 1
        redecided.append({**r, "pred_operation": b.pred_operation})
    metrics = operation_metrics(rows)
    matrix = metrics["confusion_matrix"]
    prior_counts = Counter(r.get("pred_operation") for r in rows)
    n = max(len(rows), 1)
    prior = {k: prior_counts.get(k, 0) / n for k in ("CONTINUE", "ROLLBACK_TO", "REPLAN")}
    op_acc = sum(
        1 for r in rows if r.get("pred_operation") == r.get("gold_operation")
    ) / n
    ck = _checkpoint_metrics(rows)
    return {
        "canonical_metrics": {
            **metrics,
            "operation_accuracy": op_acc,
            "ContinueRecall": _recall(matrix, "CONTINUE"),
            "RollbackRecall": _recall(matrix, "ROLLBACK_TO"),
            "prediction_prior": prior,
            **ck,
        },
        "canonical_parity": {
            "operation_agreement": 1.0 - mismatch / n,
            "mismatch": mismatch,
            "fallback": fallback,
            "disable_replan_violations": replan,
            "pass": mismatch == 0 and fallback == 0 and replan == 0,
        },
        "n": len(rows),
        "source": str(src),
    }


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

    report = {
        "variant": args.variant,
        "train": train_meta,
        "offline_valid": split_metrics(args.variant_dir, "offline_valid"),
        "holdout": split_metrics(args.variant_dir, "holdout"),
        "scorer_backend": "vllm_canonical",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant": args.variant, "wrote": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
