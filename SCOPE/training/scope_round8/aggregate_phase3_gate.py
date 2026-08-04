#!/usr/bin/env python3
"""Aggregate Phase 3 closed-loop rollback metrics and Hard-capability Gate."""

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

OUT = _REPO / "outputs/scope_round8"
PHASE3 = OUT / "phase3_closed_loop"
MAIN_SEEDS = ["rollback_o7_seed42", "rollback_o7_seed43", "rollback_o7_seed44"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_run(run_dir: Path) -> dict[str, Any]:
    episodes = load_jsonl(run_dir / "episodes.jsonl")
    events = load_jsonl(run_dir / "rollback_events.jsonl")
    recalls = [float(e.get("recall", 0.0)) for e in episodes]

    shadow_roll = sum(1 for e in events if e.get("shadow_operation") == "ROLLBACK_TO")
    pred_roll = sum(1 for e in events if e.get("student_operation") == "ROLLBACK_TO")
    false_roll = sum(
        1
        for e in events
        if e.get("student_operation") == "ROLLBACK_TO"
        and e.get("shadow_operation") != "ROLLBACK_TO"
    )
    roll_recall = sum(
        1
        for e in events
        if e.get("shadow_operation") == "ROLLBACK_TO"
        and e.get("student_operation") == "ROLLBACK_TO"
    ) / max(shadow_roll, 1)
    roll_precision = sum(
        1
        for e in events
        if e.get("student_operation") == "ROLLBACK_TO"
        and e.get("shadow_operation") == "ROLLBACK_TO"
    ) / max(pred_roll, 1)
    false_roll_rate = false_roll / max(len(events), 1)

    ck_correct = sum(
        1
        for e in events
        if e.get("shadow_operation") == "ROLLBACK_TO"
        and e.get("student_operation") == "ROLLBACK_TO"
        and e.get("predicted_checkpoint_id") == e.get("shadow_checkpoint_id")
    )
    ck_acc = ck_correct / max(shadow_roll, 1)

    continue_shadow = sum(1 for e in events if e.get("shadow_operation") == "CONTINUE")
    continue_correct = sum(
        1
        for e in events
        if e.get("shadow_operation") == "CONTINUE"
        and e.get("student_operation") == "CONTINUE"
    )
    continue_recall = continue_correct / max(continue_shadow, 1)

    restore_ok = sum(1 for e in events if e.get("state_hash_restore"))
    restore_rate = restore_ok / max(
        sum(1 for e in events if e.get("student_operation") == "ROLLBACK_TO"), 1
    )

    op_correct = sum(
        1
        for e in events
        if e.get("student_operation") == e.get("shadow_operation")
    )
    op_bal_acc = op_correct / max(len(events), 1)

    return {
        "n_episodes": len(episodes),
        "n_events": len(events),
        "mean_recall": sum(recalls) / max(len(recalls), 1),
        "RollbackRecall": roll_recall,
        "RollbackPrecision": roll_precision,
        "FalseRollbackRate": false_roll_rate,
        "ContinueRecall": continue_recall,
        "target_checkpoint_accuracy": ck_acc,
        "operation_balanced_accuracy": op_bal_acc,
        "state_hash_restore_rate": restore_rate,
        "budget_violations": sum(
            float((e.get("rollback_telemetry") or {}).get("budget_violations", 0))
            for e in episodes
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "HARD_CAPABILITY_GATE_PHASE3.json")
    args = p.parse_args()

    variants: dict[str, Any] = {}
    for child in sorted(PHASE3.iterdir()) if PHASE3.exists() else []:
        if not child.is_dir():
            continue
        shards = [child / f"shard{i}" for i in range(4)]
        merged_events: list[dict] = []
        merged_episodes: list[dict] = []
        for sh in shards:
            merged_events.extend(load_jsonl(sh / "rollback_events.jsonl"))
            merged_episodes.extend(load_jsonl(sh / "episodes.jsonl"))
        tmp = child / "_agg"
        tmp.mkdir(exist_ok=True)
        (tmp / "rollback_events.jsonl").write_text(
            "\n".join(json.dumps(r) for r in merged_events) + ("\n" if merged_events else ""),
            encoding="utf-8",
        )
        (tmp / "episodes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in merged_episodes) + ("\n" if merged_episodes else ""),
            encoding="utf-8",
        )
        variants[child.name] = aggregate_run(tmp)

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
    soft = variants.get("rollback_soft_replan_only", {})
    recovery_better = all(
        m.get("RollbackRecall", 0) > base.get("RollbackRecall", 0)
        for m in seed_metrics
        if m
    )

    report = {
        "variants": variants,
        "main_seeds_pass": seeds_pass,
        "recovery_better_than_base": recovery_better,
        "hard_capability_positive_signal": seeds_pass and recovery_better,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
