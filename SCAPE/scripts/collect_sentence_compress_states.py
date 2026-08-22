#!/usr/bin/env python3
"""Collect sentence_compress-active on-policy Student states.

Does not invent Teacher events. If --from-rollout-jsonl is omitted, this
writes the collector contract and, with --live, reuses the four-cell runtime
rollout to harvest current-policy states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.state.snapshot import EnvironmentSnapshot, capture_snapshot
from scape.adapters.components import minus_mask
from scape.training.on_policy_collector import filter_component_states, write_collected_states
from scape.training.rl_opd_types import StudentDecisionPoint
from scape.training.sentence_compress_teacher import is_compression_active_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--from-rollout-jsonl", type=Path, default=None)
    p.add_argument("--component", default="sentence_compress")
    p.add_argument("--min-chars", type=int, default=200)
    return p.parse_args()


def _point_from_row(row: dict) -> StudentDecisionPoint:
    snap_payload = row.get("pre_action_snapshot") or row.get("snapshot") or {}
    if snap_payload:
        snap = EnvironmentSnapshot.from_dict(snap_payload)
    else:
        snap = capture_snapshot(
            query_id=str(row.get("query_id") or "q"),
            step=int(row.get("turn_id") or 0),
            harness_mask=minus_mask("sentence_compress"),
            working_memory=dict(row.get("working_memory") or {}),
        )
    return StudentDecisionPoint(
        episode_id=str(row.get("episode_id") or row.get("query_id") or "ep"),
        query_id=str(row.get("query_id") or "q"),
        rollout_idx=int(row.get("rollout_idx") or 0),
        turn_id=int(row.get("turn_id") or 0),
        policy_version=str(row.get("policy_version") or "v0"),
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=str(row.get("pre_action_snapshot_hash") or snap.content_hash()),
        student_model_input=row.get("student_model_input") or "",
        student_action_tokens=list(row.get("student_action_tokens") or []),
        student_action_text=str(row.get("student_action_text") or ""),
        action_tool_names=list(row.get("action_tool_names") or []),
        structurally_valid=bool(row.get("structurally_valid", True)),
        reward=row.get("reward"),
    )


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_rollout_jsonl is None:
        payload = {
            "status": "COLLECTOR_READY",
            "component": args.component,
            "opd_state_source": "current_on_policy_rl_rollout",
            "note": "Pass --from-rollout-jsonl or run run_sr_opd_four_cell.py which writes collected_states.jsonl per cell.",
        }
        (args.out / "COLLECTION_CONTRACT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    points = []
    with args.from_rollout_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                points.append(_point_from_row(json.loads(line)))
    kept = [
        p
        for p in filter_component_states(points, component_id=args.component, require_valid=False)
        if is_compression_active_state(p.pre_action_snapshot.working_memory, min_chars=args.min_chars)
        or args.component != "sentence_compress"
    ]
    audit = write_collected_states(kept, args.out / "collected_states.jsonl", component_id=args.component)
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
