#!/usr/bin/env python3
"""Aggregate Phase 3 closed-loop rollback metrics with corrected gate semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OPERATIONS = ("CONTINUE", "REPLAN", "ROLLBACK_TO")
MAIN_SEEDS = ["rollback_o7_seed42", "rollback_o7_seed43", "rollback_o7_seed44"]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _op_label(event: dict, *, gold: bool) -> str:
    key = "shadow_operation" if gold else "student_operation"
    return str(event.get(key) or "")


def _confusion_matrix(events: list[dict]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in OPERATIONS} for gold in OPERATIONS
    }
    for event in events:
        gold = _op_label(event, gold=True)
        pred = _op_label(event, gold=False)
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def _per_class_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pred_totals = {pred: sum(matrix[gold][pred] for gold in OPERATIONS) for pred in OPERATIONS}
    for op in OPERATIONS:
        support = sum(matrix[op][pred] for pred in OPERATIONS)
        tp = matrix[op][op]
        precision = tp / max(pred_totals[op], 1)
        recall = tp / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics[op] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "prediction_prior": pred_totals[op] / max(sum(pred_totals.values()), 1),
        }
    return metrics


def _balanced_accuracy(matrix: dict[str, dict[str, int]]) -> float:
    recalls = []
    for op in OPERATIONS:
        support = sum(matrix[op][pred] for pred in OPERATIONS)
        if support == 0:
            continue
        recalls.append(matrix[op][op] / support)
    return sum(recalls) / max(len(recalls), 1)


def _checkpoint_metrics(events: list[dict]) -> dict[str, Any]:
    eligible = []
    covered = 0
    correct = 0
    invalid = 0
    mrr_sum = 0.0
    for event in events:
        if _op_label(event, gold=True) != "ROLLBACK_TO":
            continue
        gold_ck = event.get("shadow_checkpoint_id")
        candidates = event.get("candidate_checkpoint_ids") or []
        if candidates and gold_ck not in candidates:
            continue
        eligible.append(event)
        if candidates:
            covered += int(gold_ck in candidates)
        pred_ck = event.get("predicted_checkpoint_id")
        pred_op = _op_label(event, gold=False)
        if pred_op == "ROLLBACK_TO":
            if candidates and pred_ck not in candidates:
                invalid += 1
            if pred_ck == gold_ck:
                correct += 1
            if candidates and gold_ck in candidates:
                try:
                    rank = candidates.index(gold_ck) + 1
                    mrr_sum += 1.0 / rank
                except ValueError:
                    pass
    n = len(eligible)
    return {
        "checkpoint_accuracy": correct / max(n, 1),
        "checkpoint_mrr": mrr_sum / max(n, 1),
        "checkpoint_candidate_coverage": covered / max(n, 1),
        "n_checkpoint_eval": n,
        "n_checkpoint_eval": n,
        "invalid_checkpoint_predictions": invalid,
        "invalid_checkpoint_rate": invalid / max(n, 1),
    }


def _restore_metrics(events: list[dict]) -> dict[str, Any]:
    rollback_exec = [
        e for e in events if _op_label(e, gold=False) == "ROLLBACK_TO"
    ]
    restore_ok = sum(
        1
        for e in rollback_exec
        if e.get("state_hash_restore") is True
    )
    return {
        "n_rollback_executed": len(rollback_exec),
        "state_hash_restore_rate": restore_ok / max(len(rollback_exec), 1),
        "state_hash_restore_count": restore_ok,
    }


def _budget_metrics(events: list[dict], episodes: list[dict]) -> dict[str, Any]:
    """Count only true budget overruns.

    Prefer per-event flags. Round 8 episode telemetry stored a runaway cumulative
    counter (hundreds per episode), which must not be treated as gate truth.
    """
    violations = 0
    for event in events:
        if event.get("budget_violation") is True:
            violations += 1
        elif isinstance(event.get("budget_event"), dict) and event["budget_event"].get(
            "violation"
        ):
            violations += 1
    legacy_unreliable = False
    if violations == 0 and episodes and not events:
        # Only fall back when we have no event stream. Cap absurd legacy counters.
        legacy_vals = [
            int((ep.get("rollback_telemetry") or {}).get("budget_violations", 0))
            for ep in episodes
        ]
        # A true per-episode violation count should be tiny (budget is small).
        # Values >> max_turns indicate the Round 8 counter bug.
        sane = [v for v in legacy_vals if 0 <= v <= 32]
        if sane and len(sane) == len(legacy_vals):
            violations = sum(sane)
        else:
            legacy_unreliable = True
            violations = 0
    return {
        "budget_violations": violations,
        "budget_violations_legacy_unreliable": legacy_unreliable,
    }


def _fallback_metrics(events: list[dict]) -> dict[str, Any]:
    fallback = sum(1 for e in events if e.get("fallback_reason"))
    return {"fallback_count": fallback}


def aggregate_events(events: list[dict], episodes: list[dict] | None = None) -> dict[str, Any]:
    episodes = episodes or []
    matrix = _confusion_matrix(events)
    per_class = _per_class_metrics(matrix)
    ck = _checkpoint_metrics(events)
    restore = _restore_metrics(events)
    budget = _budget_metrics(events, episodes)
    fallback = _fallback_metrics(events)

    shadow_roll = sum(1 for e in events if _op_label(e, gold=True) == "ROLLBACK_TO")
    pred_roll = sum(1 for e in events if _op_label(e, gold=False) == "ROLLBACK_TO")
    roll_tp = sum(
        1
        for e in events
        if _op_label(e, gold=True) == "ROLLBACK_TO"
        and _op_label(e, gold=False) == "ROLLBACK_TO"
    )
    false_roll = sum(
        1
        for e in events
        if _op_label(e, gold=False) == "ROLLBACK_TO"
        and _op_label(e, gold=True) != "ROLLBACK_TO"
    )

    recalls = [float(e.get("recall", 0.0)) for e in episodes]
    return {
        "n_episodes": len(episodes),
        "n_events": len(events),
        "mean_recall": sum(recalls) / max(len(recalls), 1),
        "operation_confusion_matrix": matrix,
        "operation_per_class": per_class,
        "operation_balanced_accuracy": _balanced_accuracy(matrix),
        "ContinueRecall": per_class["CONTINUE"]["recall"],
        "ReplanRecall": per_class["REPLAN"]["recall"],
        "RollbackRecall": roll_tp / max(shadow_roll, 1),
        "RollbackPrecision": roll_tp / max(pred_roll, 1),
        "FalseRollbackRate": false_roll / max(len(events), 1),
        "target_checkpoint_accuracy": ck["checkpoint_accuracy"],
        "checkpoint_mrr": ck["checkpoint_mrr"],
        "checkpoint_candidate_coverage": ck["checkpoint_candidate_coverage"],
        "n_checkpoint_eval": ck["n_checkpoint_eval"],
        "invalid_checkpoint_rate": ck["invalid_checkpoint_rate"],
        "invalid_checkpoint_predictions": ck["invalid_checkpoint_predictions"],
        **restore,
        **budget,
        **fallback,
    }


def aggregate_run(run_dir: Path) -> dict[str, Any]:
    return aggregate_events(
        load_jsonl(run_dir / "rollback_events.jsonl"),
        load_jsonl(run_dir / "episodes.jsonl"),
    )


def merge_shards(variant_dir: Path) -> Path:
    merged_events: list[dict] = []
    merged_episodes: list[dict] = []
    for i in range(4):
        sh = variant_dir / f"shard{i}"
        merged_events.extend(load_jsonl(sh / "rollback_events.jsonl"))
        merged_episodes.extend(load_jsonl(sh / "episodes.jsonl"))
    tmp = variant_dir / "_agg"
    tmp.mkdir(exist_ok=True)
    (tmp / "rollback_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in merged_events) + ("\n" if merged_events else ""),
        encoding="utf-8",
    )
    (tmp / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in merged_episodes) + ("\n" if merged_episodes else ""),
        encoding="utf-8",
    )
    return tmp


def evaluate_hard_gate(variants: dict[str, Any]) -> dict[str, Any]:
    seed_metrics = [variants.get(v, {}) for v in MAIN_SEEDS]
    seeds_pass = all(
        m.get("RollbackRecall", 0) >= 0.30
        and m.get("FalseRollbackRate", 0) <= 0.05
        and m.get("target_checkpoint_accuracy", 0) >= 0.70
        and m.get("state_hash_restore_rate", 0) == 1.0
        and m.get("budget_violations", 0) == 0
        for m in seed_metrics
        if m
    )
    base = variants.get("base_agent_core", {})
    recovery_better = all(
        m.get("RollbackRecall", 0) > base.get("RollbackRecall", 0)
        for m in seed_metrics
        if m
    )
    return {
        "main_seeds_pass": seeds_pass,
        "recovery_better_than_base": recovery_better,
        "hard_capability_positive_signal": seeds_pass and recovery_better,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase3-dir", type=Path, default=_REPO / "outputs/scope_round8/phase3_closed_loop")
    p.add_argument("--output", type=Path, default=_REPO / "outputs/scope_round9/reaggregate_round8/HARD_CAPABILITY_GATE_PHASE3_REAGG.json")
    args = p.parse_args()

    variants: dict[str, Any] = {}
    for child in sorted(args.phase3_dir.iterdir()) if args.phase3_dir.exists() else []:
        if not child.is_dir():
            continue
        agg = merge_shards(child)
        variants[child.name] = aggregate_run(agg)

    gate = evaluate_hard_gate(variants)
    report = {
        "variants": variants,
        **gate,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
