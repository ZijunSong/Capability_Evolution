"""Teacher side-branch for the Teacher-only ``verify`` tool.

Do not invent ``verified=True``. Without a real verifier result this branch
skips. The ablation path may emit a student-legal follow-up without claiming
a verification outcome.
"""

from __future__ import annotations

from typing import Any, Mapping

from trim.training.opd_events import HarnessEvent, model_action
from trim.training.rl_opd_types import StudentDecisionPoint
from trim.training.teacher_branch import (
    SOURCE_SKIP_UNTRIGGERED,
    SOURCE_SYNTHETIC,
    filter_unworthy_events,
    skip_teacher_events,
    tag_source,
)

COMPONENT_ID = "verify_tool"


def _visible_doc_ids(wm: Mapping[str, Any]) -> list[str]:
    ids = [str(x) for x in (wm.get("curated_ids") or []) if str(x)]
    if ids:
        return ids[:4]
    docs = wm.get("documents") or []
    for rec in docs:
        if isinstance(rec, Mapping):
            did = rec.get("id") or rec.get("doc_id")
            if did is not None and str(did):
                ids.append(str(did))
    return ids[:4]


def teacher_events_from_wm(
    wm: Mapping[str, Any],
    *,
    turn_id: int = 0,
    query: str | None = None,
    allow_synthetic_heuristic: bool = False,
) -> list[HarnessEvent]:
    del query
    if not allow_synthetic_heuristic:
        return skip_teacher_events(
            COMPONENT_ID,
            turn_id=turn_id,
            reason="no_real_verifier",
            teacher_kind=SOURCE_SKIP_UNTRIGGERED,
        )
    doc_ids = _visible_doc_ids(wm)
    if not doc_ids:
        return skip_teacher_events(
            COMPONENT_ID,
            turn_id=turn_id,
            reason="no_evidence_ids",
            teacher_kind=SOURCE_SKIP_UNTRIGGERED,
        )
    events = tag_source(
        [
            model_action(
                "read_document",
                {"doc_id": doc_ids[0]},
                turn_id=turn_id,
                component_id=COMPONENT_ID,
                metadata={"student_realizable": True, "synthetic_heuristic": True},
            )
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
