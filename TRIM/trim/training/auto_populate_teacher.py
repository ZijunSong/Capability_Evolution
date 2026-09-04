"""Teacher side-branch for auto_populate_first_search.

The full Harness automatically copies the first search pool into curated.  The
student cannot see that hidden mutation, so SR-OPD exposes only the equivalent
student-executable curated delta and never the full-only view.
"""

from __future__ import annotations

from typing import Any, Mapping

from trim.training.opd_events import HarnessEvent, harness_mutation, model_action
from trim.training.rl_opd_types import StudentDecisionPoint

COMPONENT_ID = "auto_populate_first_search"


def _records(wm: Mapping[str, Any]) -> dict[str, Any]:
    pool = wm.get("pool") or {}
    return dict(pool) if isinstance(pool, dict) else {}


def teacher_events_from_wm(
    wm: Mapping[str, Any], *, turn_id: int = 0, query: str | None = None
) -> list[HarnessEvent]:
    """Emit the hidden auto mutation, or the first-search anchor if pending."""
    pool = _records(wm)
    curated = [str(x) for x in (wm.get("curated_ids") or [])]
    if pool:
        ranked = sorted(
            pool.items(),
            key=lambda item: (-float((item[1] or {}).get("score") or 0.0), str(item[0])),
        )
        added = [str(did) for did, _ in ranked[:8] if str(did) not in set(curated)]
        after = list(dict.fromkeys(curated + added))
        if after != curated:
            return [
                harness_mutation(
                    COMPONENT_ID,
                    {"before_curated": curated, "after_curated": after},
                    turn_id=turn_id,
                    metadata={"owner": "teacher_full", "hidden_auto_effect": True},
                )
            ]
    q = str(query if query is not None else wm.get("query") or "")
    return [
        model_action(
            "search_corpus",
            {"query": q},
            turn_id=turn_id,
            component_id=COMPONENT_ID,
            metadata={"owner": "teacher_full", "auto_anchor": True},
        )
    ]


def teacher_events_from_point(point: StudentDecisionPoint) -> list[HarnessEvent]:
    wm = point.pre_action_snapshot.working_memory
    return teacher_events_from_wm(wm, turn_id=int(point.turn_id), query=wm.get("query"))
