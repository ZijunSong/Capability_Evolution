#!/usr/bin/env python3
"""Capability-level Dup evaluation with operation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.capability.dup_operation import DupOperation
from training.scope.binary_operation_metrics import accumulate_from_pairs
from training.scope.compact_target import (
    compact_target_from_sample,
    infer_operation_from_action,
)
from training.scope.dup_diagnostics import load_jsonl
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def _parse_operation(pred: dict[str, Any] | None) -> DupOperation | None:
    if not pred:
        return None
    if "operation" in pred:
        try:
            return DupOperation(str(pred["operation"]).upper())
        except ValueError:
            return None
    return infer_operation_from_action(pred)


def _argument_set_f1(pred: dict[str, Any] | None, tgt: dict[str, Any] | None) -> float:
    if not pred or not tgt:
        return 0.0
    pred_keys = {k for k in ("candidate_id", "canonical_id") if pred.get(k)}
    tgt_keys = {k for k in ("candidate_id", "canonical_id") if tgt.get(k)}
    if not tgt_keys:
        return 1.0
    inter = sum(1 for k in tgt_keys if pred.get(k) == tgt.get(k))
    prec = inter / max(len(pred_keys), 1)
    rec = inter / max(len(tgt_keys), 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)


def evaluate_capability(
    trainer: DupSDITrainer,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    trainer.model.eval()
    n = len(samples)
    op_correct = 0
    exact_match = 0
    parse_ok = 0
    arg_f1_sum = 0.0
    confusion: dict[str, Counter] = defaultdict(Counter)
    route_op_correct: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    pairs: list[tuple[DupOperation | None, DupOperation | None]] = []

    for sample in samples:
        pred = trainer._greedy_action(sample)
        tgt_action = sample.get("target_action")
        compact = compact_target_from_sample(sample)
        pred_op = _parse_operation(pred)
        tgt_op = compact.operation if compact else _parse_operation(tgt_action)

        if pred is not None:
            parse_ok += 1
        if pred_op == tgt_op and tgt_op is not None:
            op_correct += 1
        if tgt_op:
            confusion[tgt_op.value][pred_op.value if pred_op else "PARSE_FAIL"] += 1
            pairs.append((tgt_op, pred_op))
        route = str(sample.get("route", "")).upper()
        if tgt_op:
            route_op_correct[route][1] += 1
            if pred_op == tgt_op:
                route_op_correct[route][0] += 1

        tgt_dict = compact.to_dict() if compact else (tgt_action or {})
        arg_f1_sum += _argument_set_f1(pred, tgt_dict)

        pred_norm = trainer._action_dict(pred) if pred else None
        tgt_norm = trainer._action_dict(tgt_action)
        if pred_norm == tgt_norm:
            exact_match += 1

    route_acc = {
        r: hits / max(total, 1)
        for r, (hits, total) in route_op_correct.items()
    }

    bin_metrics = accumulate_from_pairs(pairs)
    metrics_dict = bin_metrics.to_dict()

    return {
        "n_valid": n,
        "operation_accuracy": op_correct / max(n, 1),
        "operation_confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        **metrics_dict,
        "argument_accuracy": arg_f1_sum / max(n, 1),
        "argument_set_f1": arg_f1_sum / max(n, 1),
        "parse_rate": parse_ok / max(n, 1),
        "exact_action_match": exact_match / max(n, 1),
        "route_operation_accuracy": route_acc,
        "teacher_forced_token_acc": None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--include-legacy-metrics", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.valid)
    cfg = SDITrainConfig(
        model_path=args.model_path,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        eval_only=True,
        output_dir=Path("/tmp/dup_eval"),
    )
    trainer = DupSDITrainer(cfg)
    report = evaluate_capability(trainer, rows)
    if args.include_legacy_metrics:
        legacy = trainer.evaluate(args.valid)
        report["teacher_forced_token_acc"] = legacy.get("teacher_forced_token_acc")
        report["action_match_rate"] = legacy.get("action_match_rate")
    report["capability"] = "duplicate_evidence"
    route_counts = Counter(str(r.get("route", "")).upper() for r in rows)
    report["route_counts"] = dict(route_counts)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
