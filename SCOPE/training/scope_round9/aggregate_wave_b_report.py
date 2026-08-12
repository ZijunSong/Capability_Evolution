#!/usr/bin/env python3
"""Merge Wave B train report with offline/holdout HF↔vLLM eval parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_frozen_replay import (
    barrier_a_for_parity,
    compare_hf_vllm,
    load_jsonl,
    operation_metrics,
)
from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy, _confusion_matrix, _per_class_metrics


def _recall(matrix: dict, op: str) -> float | None:
    """Per-class recall; None when gold support is 0 (do not veto the gate)."""
    support = sum(matrix.get(op, {}).values()) if isinstance(matrix.get(op), dict) else 0
    if support <= 0:
        return None
    return matrix[op][op] / support


def split_report(variant_dir: Path, split: str) -> dict:
    hf_path = variant_dir / f"eval_{split}" / "hf_replay.jsonl"
    vllm_path = variant_dir / f"eval_{split}" / "vllm_replay.jsonl"
    if not hf_path.exists():
        return {}
    hf_rows = load_jsonl(hf_path)
    vllm_rows = load_jsonl(vllm_path) if vllm_path.exists() else []
    parity = compare_hf_vllm(hf_rows, vllm_rows) if vllm_rows else {}
    metrics = operation_metrics(hf_rows)
    matrix = metrics["confusion_matrix"]
    per_class = _per_class_metrics(matrix)
    return {
        "hf_metrics": {
            **metrics,
            "ContinueRecall": _recall(matrix, "CONTINUE"),
            "ReplanRecall": _recall(matrix, "REPLAN"),
            "RollbackRecall": _recall(matrix, "ROLLBACK_TO"),
            "per_class": per_class,
        },
        "parity": parity,
        "parity_pass": barrier_a_for_parity(parity)[0] if parity else False,
    }


def offline_gate(main_seed_reports: list[dict]) -> dict:
    """Evaluate Proposed Offline Gate on main hierarchical seeds."""
    checks = []
    for r in main_seed_reports:
        off = r.get("offline_valid", {}).get("hf_metrics", {})
        hold = r.get("holdout", {}).get("hf_metrics", {})
        parity_ok = r.get("offline_valid", {}).get("parity_pass") and r.get("holdout", {}).get(
            "parity_pass"
        )
        def _req_recall(val, floor: float = 0.70) -> bool:
            # Missing / n/a (no gold support) does not fail the gate.
            return val is None or float(val) >= floor

        row = {
            "variant": r.get("variant"),
            "operation_balanced_accuracy": off.get("operation_balanced_accuracy", 0),
            "ContinueRecall": off.get("ContinueRecall"),
            "ReplanRecall": off.get("ReplanRecall"),
            "RollbackRecall": off.get("RollbackRecall"),
            "holdout_operation_balanced_accuracy": hold.get("operation_balanced_accuracy", 0),
            "holdout_ContinueRecall": hold.get("ContinueRecall"),
            "parity_ok": parity_ok,
        }
        row["pass"] = (
            row["operation_balanced_accuracy"] >= 0.80
            and _req_recall(row["ContinueRecall"])
            and _req_recall(row["ReplanRecall"])
            and _req_recall(row["RollbackRecall"])
            and row["holdout_operation_balanced_accuracy"] >= 0.70
            and _req_recall(row["holdout_ContinueRecall"])
            and bool(parity_ok)
        )
        checks.append(row)
    return {
        "main_seed_checks": checks,
        "offline_gate_pass": bool(checks) and all(c["pass"] for c in checks),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    train_meta = {}
    # run_wave_b_train writes TRAIN_AND_EVAL_REPORT.json before this script; relocate if needed.
    existing = args.variant_dir / "TRAIN_AND_EVAL_REPORT.json"
    if existing.exists():
        try:
            train_meta = json.loads(existing.read_text(encoding="utf-8"))
        except Exception:
            train_meta = {}
        existing.rename(args.variant_dir / "train_only_report.json")

    report = {
        "variant": args.variant,
        "train": train_meta,
        "offline_valid": split_report(args.variant_dir, "offline_valid"),
        "holdout": split_report(args.variant_dir, "holdout"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# Wave B report: {args.variant}",
        "",
        f"- offline bal_acc: {report['offline_valid'].get('hf_metrics', {}).get('operation_balanced_accuracy')}",
        f"- holdout bal_acc: {report['holdout'].get('hf_metrics', {}).get('operation_balanced_accuracy')}",
        f"- offline parity pass: {report['offline_valid'].get('parity_pass')}",
        f"- holdout parity pass: {report['holdout'].get('parity_pass')}",
        "",
    ]
    (args.variant_dir / "TRAIN_AND_EVAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
