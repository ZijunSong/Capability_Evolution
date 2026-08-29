"""Teacher side-branch for the Teacher-only ``verify`` tool.

The verification result is hidden from the Student. ``verify`` itself is ε:
skip-to-anchor keeps scanning until a Student-native tool call appears.
This helper still emits a subsequent ``read_document`` only when the Teacher
trace has no later recorded Student action; the projector never invents a
recovery macro of its own.
"""

from __future__ import annotations

from typing import Any, Mapping

from scape.training.opd_events import HarnessEvent, model_action, obs_transform
from scape.training.rl_opd_types import StudentDecisionPoint

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
    wm: Mapping[str, Any], *, turn_id: int = 0, query: str | None = None
) -> list[HarnessEvent]:
    q = str(query if query is not None else wm.get("query") or "")
    doc_ids = _visible_doc_ids(wm)
    if not doc_ids:
        return [model_action("search_corpus", {"query": q}, turn_id=turn_id, component_id=COMPONENT_ID)]
    return [
        obs_transform(
            COMPONENT_ID,
            turn_id=turn_id,
            observation={"verified": True, "evidence_ids": doc_ids},
            visible_to_student=False,
            metadata={"owner": "teacher_full", "student_must_not_see": True},
        ),
        model_action(
            "verify",
            {"evidence_ids": doc_ids, "claim": q[:512]},
            turn_id=turn_id,
            component_id=COMPONENT_ID,
            visible_to_student=False,
            metadata={"teacher_only": True},
        ),
        model_action(
            "read_document",
            {"doc_id": doc_ids[0]},
            turn_id=turn_id,
            component_id=COMPONENT_ID,
            metadata={"student_realizable": True},
        ),
    ]


def teacher_events_from_point(point: StudentDecisionPoint) -> list[HarnessEvent]:
    wm = point.pre_action_snapshot.working_memory
    return teacher_events_from_wm(wm, turn_id=int(point.turn_id), query=wm.get("query"))
