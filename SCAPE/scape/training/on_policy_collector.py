"""On-policy Student decision collector.

Consumes live or recorded Reduced-Harness rollouts. It never invents
Teacher events or mutates environment rewards.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from scape.training.rl_opd_types import StudentDecisionPoint
from scape.training.sentence_compress_teacher import is_compression_active_state

COMPONENT_FILTERS = {
    "sentence_compress": lambda point: is_compression_active_state(point.pre_action_snapshot.working_memory),
}


def component_active(point: StudentDecisionPoint, component_id: str) -> bool:
    filt = COMPONENT_FILTERS.get(component_id)
    if filt is None:
        return True
    return bool(filt(point))


def filter_component_states(
    points: Iterable[StudentDecisionPoint],
    *,
    component_id: str,
    require_valid: bool = True,
) -> list[StudentDecisionPoint]:
    kept: list[StudentDecisionPoint] = []
    for point in points:
        if require_valid and not point.structurally_valid:
            continue
        if component_active(point, component_id):
            kept.append(point)
    return kept


def decision_point_to_row(point: StudentDecisionPoint) -> dict[str, Any]:
    wm = point.pre_action_snapshot.working_memory
    return {
        "decision_point_id": point.decision_point_id,
        "episode_id": point.episode_id,
        "query_id": point.query_id,
        "rollout_idx": point.rollout_idx,
        "turn_id": point.turn_id,
        "policy_version": point.policy_version,
        "pre_action_snapshot": point.pre_action_snapshot.to_dict(),
        "pre_action_snapshot_hash": point.pre_action_snapshot_hash,
        "student_action_text": point.student_action_text,
        "action_tool_names": list(point.action_tool_names),
        "structurally_valid": point.structurally_valid,
        "reward": point.reward,
        "visible_doc_count": len(wm.get("documents") or wm.get("pool") or {}),
        "curated_ids": list(wm.get("curated_ids") or []),
        "query": wm.get("query"),
    }


def write_collected_states(
    points: Sequence[StudentDecisionPoint],
    path: Path,
    *,
    component_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [decision_point_to_row(p) for p in points]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "component_id": component_id,
        "n_states": len(rows),
        "n_queries": len({r["query_id"] for r in rows}),
        "path": str(path),
        "opd_state_source": "current_on_policy_rl_rollout",
        **dict(extra or {}),
    }
    path.with_name("COLLECTION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit
