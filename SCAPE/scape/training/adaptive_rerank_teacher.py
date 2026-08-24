"""Teacher side-branch for adaptive_rerank_instruction.

The rerank instruction is teacher-only context.  The emitted search action uses
only the Student-visible query, so SR-OPD can project it as a direct,
Student-realizable target without copying ranked results or hidden metadata.
"""

from __future__ import annotations

from typing import Any, Mapping

from scape.training.opd_events import HarnessEvent, model_action, obs_transform
from scape.training.rl_opd_types import StudentDecisionPoint

COMPONENT_ID = "adaptive_rerank_instruction"
RERANK_INSTRUCTION_KEY = "rerank_instruction"


def teacher_events_from_wm(
    wm: Mapping[str, Any], *, turn_id: int = 0, query: str | None = None
) -> list[HarnessEvent]:
    q = str(query if query is not None else wm.get("query") or "")
    instruction = str(
        wm.get(RERANK_INSTRUCTION_KEY)
        or "prefer direct evidence and diverse corroborating sources"
    )
    return [
        obs_transform(
            COMPONENT_ID,
            turn_id=turn_id,
            observation={RERANK_INSTRUCTION_KEY: instruction},
            visible_to_student=False,
            metadata={
                "owner": "teacher_full",
                "student_must_not_see": True,
                "student_realizable": False,
            },
        ),
        # Query-only action: no teacher result IDs or rewritten query are copied.
        model_action(
            "search_corpus",
            {"query": q},
            turn_id=turn_id,
            component_id=COMPONENT_ID,
            metadata={"student_realizable": True, "teacher_result_ids": []},
        ),
    ]


def teacher_events_from_point(point: StudentDecisionPoint) -> list[HarnessEvent]:
    wm = point.pre_action_snapshot.working_memory
    return teacher_events_from_wm(wm, turn_id=int(point.turn_id), query=wm.get("query"))
