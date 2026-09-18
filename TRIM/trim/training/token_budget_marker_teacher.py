"""Teacher side-branch for token_budget_marker SR-OPD projection.

An empty marker is untriggered. A present marker is still only accounting
context; the default path does not invent a search/read target. The ablation
path may emit the previous heuristic action.
"""

from __future__ import annotations

from typing import Any, Mapping

from trim.training.opd_events import HarnessEvent, model_action, obs_transform
from trim.training.rl_opd_types import StudentDecisionPoint
from trim.training.teacher_branch import (
    SOURCE_SKIP_UNTRIGGERED,
    SOURCE_SYNTHETIC,
    filter_unworthy_events,
    skip_teacher_events,
    tag_source,
)

COMPONENT_ID = "token_budget_marker"


def _next_action(wm: Mapping[str, Any], query: str) -> tuple[str, dict[str, Any]]:
    docs = list(wm.get("documents") or [])
    if docs:
        first = docs[0] if isinstance(docs[0], dict) else {"id": docs[0]}
        doc_id = str(first.get("id") or first.get("doc_id") or "")
        if doc_id:
            return "read_document", {"doc_id": doc_id}
    return "search_corpus", {"query": query}


def teacher_events_from_wm(
    wm: Mapping[str, Any],
    *,
    turn_id: int = 0,
    query: str | None = None,
    allow_synthetic_heuristic: bool = False,
) -> list[HarnessEvent]:
    q = str(query if query is not None else wm.get("query") or "")
    marker = str(wm.get("token_budget_marker") or "")
    if not marker.strip():
        return skip_teacher_events(
            COMPONENT_ID,
            turn_id=turn_id,
            reason="empty_token_budget_marker",
            teacher_kind=SOURCE_SKIP_UNTRIGGERED,
        )
    if not allow_synthetic_heuristic:
        return skip_teacher_events(
            COMPONENT_ID,
            turn_id=turn_id,
            reason="no_capability_derived_action",
            teacher_kind=SOURCE_SKIP_UNTRIGGERED,
        )
    name, arguments = _next_action(wm, q)
    events = tag_source(
        [
            obs_transform(
                COMPONENT_ID,
                turn_id=turn_id,
                observation={"token_budget_marker": marker},
                visible_to_student=False,
                metadata={
                    "owner": "teacher_full",
                    "student_must_not_see": True,
                    "requires_external_accounting": False,
                },
            ),
            model_action(name, arguments, turn_id=turn_id, component_id=COMPONENT_ID),
        ],
        SOURCE_SYNTHETIC,
    )
    return filter_unworthy_events(wm, events, component_id=COMPONENT_ID, turn_id=turn_id)


def teacher_events_from_point(
    point: StudentDecisionPoint,
    *,
    allow_synthetic_heuristic: bool = False,
) -> list[HarnessEvent]:
    wm = point.pre_action_snapshot.working_memory
    return teacher_events_from_wm(
        wm,
        turn_id=int(point.turn_id),
        query=wm.get("query"),
        allow_synthetic_heuristic=allow_synthetic_heuristic,
    )
