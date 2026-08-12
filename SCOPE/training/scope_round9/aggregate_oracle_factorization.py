#!/usr/bin/env python3
"""Oracle factorization on frozen base_live states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy, _checkpoint_metrics, _confusion_matrix


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_mode(row: dict, mode: str) -> dict:
    pred_op = row.get("pred_operation")
    pred_ck = row.get("pred_checkpoint_global_id")
    gold_op = row.get("gold_operation")
    gold_ck = row.get("gold_checkpoint_global_id")
    if mode.startswith("oracle_op"):
        pred_op = gold_op
    if mode.endswith("oracle_ckpt"):
        pred_ck = gold_ck
    return {
        "shadow_operation": gold_op,
        "student_operation": pred_op,
        "shadow_checkpoint_id": gold_ck,
        "predicted_checkpoint_id": pred_ck,
        "candidate_checkpoint_ids": [
            c.get("checkpoint_id") for c in (row.get("candidate_list") or [])
        ],
    }


def eval_mode(rows: list[dict], mode: str) -> dict:
    events = [apply_mode(r, mode) for r in rows]
    matrix = _confusion_matrix(events)
    ck = _checkpoint_metrics(events)
    per_class = {op: matrix[op][op] / max(sum(matrix[op].values()), 1) for op in matrix}
    return {
        "mode": mode,
        "operation_balanced_accuracy": _balanced_accuracy(matrix),
        "ContinueRecall": per_class.get("CONTINUE", 0.0),
        "RollbackRecall": per_class.get("ROLLBACK_TO", 0.0),
        "checkpoint_top1": ck["checkpoint_accuracy"],
        "checkpoint_mrr": ck["checkpoint_mrr"],
        "candidate_coverage": ck["checkpoint_candidate_coverage"],
        "n_events": len(events),
    }


MODES = [
    "learned_op + learned_ckpt",
    "oracle_op + learned_ckpt",
    "learned_op + oracle_ckpt",
    "oracle_op + oracle_ckpt",
]


def diagnose(results: dict[str, dict]) -> dict:
    full = results["learned_op + learned_ckpt"]
    op_only = results["learned_op + oracle_ckpt"]
    ck_only = results["oracle_op + learned_ckpt"]
    upper = results["oracle_op + oracle_ckpt"]
    bottlenecks = []
    if op_only["operation_balanced_accuracy"] < 0.5:
        bottlenecks.append("operation_classifier_live_distribution")
    if ck_only["checkpoint_top1"] < 0.5:
        bottlenecks.append("checkpoint_selector_representation")
    if (
        op_only["operation_balanced_accuracy"] >= 0.7
        and ck_only["checkpoint_top1"] >= 0.7
        and full["operation_balanced_accuracy"] < 0.5
    ):
        bottlenecks.append("action_realization_or_combination_interface")
    if upper["operation_balanced_accuracy"] < 0.99 or upper["checkpoint_top1"] < 0.99:
        bottlenecks.append("labels_candidates_or_aggregator")
    return {
        "primary_bottlenecks": bottlenecks,
        "learned_full_operation_bal_acc": full["operation_balanced_accuracy"],
        "learned_op_oracle_ckpt_bal_acc": op_only["operation_balanced_accuracy"],
        "oracle_op_learned_ckpt_top1": ck_only["checkpoint_top1"],
        "oracle_upper_bound_operation": upper["operation_balanced_accuracy"],
        "oracle_upper_bound_checkpoint": upper["checkpoint_top1"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True, help="HF replay on base_live")
    p.add_argument("--output", type=Path, default=_REPO / "outputs/scope_round9/diagnosis/ROOT_CAUSE_DECISION.json")
    args = p.parse_args()

    rows = load_jsonl(args.replay)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r.get("state_source", "base_live"), []).append(r)

    report: dict = {"modes": {}, "by_state_source": {}, "diagnosis": {}}
    for source, subset in by_source.items():
        mode_results = {m: eval_mode(subset, m) for m in MODES}
        report["by_state_source"][source] = mode_results
        if source == "base_live":
            report["modes"] = mode_results
            report["diagnosis"] = diagnose(mode_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
