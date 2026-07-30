#!/usr/bin/env python3
"""Aggregate closed-loop episodes from jsonl (Round 6 gate metrics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.dup_telemetry import AdmissionEvent, DupTelemetryAggregator
from training.scope_round6.common import load_jsonl, write_json
from training.scope_round6.metrics import direct_behavior_metrics


KEEP = DupOperation.KEEP_EVIDENCE.value
SKIP = DupOperation.SKIP_DUPLICATE.value


def aggregate_from_jsonl(episodes_path: Path, events_path: Path) -> dict[str, Any]:
    episodes = load_jsonl(episodes_path) if episodes_path.exists() else []
    events_raw = load_jsonl(events_path) if events_path.exists() else []

    tel = DupTelemetryAggregator()
    labels: list[str] = []
    preds: list[str] = []
    for ev in events_raw:
        tel.add(AdmissionEvent(
            candidate_evidence_id=str(ev.get("candidate_evidence_id", "")),
            candidate_is_duplicate=bool(ev.get("candidate_is_duplicate")),
            student_operation=ev.get("student_operation"),
            shadow_operation=ev.get("shadow_operation"),
            route=ev.get("route"),
            realized_runtime_action=ev.get("realized_runtime_action"),
            actually_curated=bool(ev.get("actually_curated")),
            query_id=str(ev.get("query_id", "")),
            turn_id=int(ev.get("turn_id", 0)),
        ))
        shadow = str(ev.get("shadow_operation") or "").upper()
        if shadow not in (KEEP, SKIP):
            shadow = SKIP if ev.get("candidate_is_duplicate") else KEEP
        pred = str(ev.get("student_operation") or "").upper()
        labels.append(shadow)
        preds.append(pred if pred in (KEEP, SKIP) else KEEP)

    summary = tel.summarize()
    direct = direct_behavior_metrics(labels, preds) if labels else {}

    rewards = [float(e.get("reward", 0)) for e in episodes]
    recalls = [float(e.get("recall", 0)) for e in episodes]
    n_curated = [float(e.get("n_curated", 0)) for e in episodes if "n_curated" in e]

    return {
        "n_episodes": len(episodes),
        "n_admission_events": len(events_raw),
        "dup_telemetry": summary,
        "direct_behavior": direct,
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "mean_recall": sum(recalls) / max(len(recalls), 1),
        "mean_n_curated": sum(n_curated) / max(len(n_curated), 1) if n_curated else 0,
        "DCR": summary.get("duplicate_curate_rate", 0),
        "FSR": summary.get("false_skip_rate", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    report = aggregate_from_jsonl(
        args.run_dir / "episodes.jsonl",
        args.run_dir / "dup_admission_events.jsonl",
    )
    out = args.output or args.run_dir / "aggregated_metrics.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
