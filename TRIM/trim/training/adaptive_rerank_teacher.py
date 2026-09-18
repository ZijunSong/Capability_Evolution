"""Teacher side-branch for adaptive_rerank_instruction.

The rerank instruction is teacher-only context. Searching the original query
does not use the capability, so the default path skips that synthetic target.
Pass ``allow_synthetic_heuristic=True`` for the low-cost ablation.
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

COMPONENT_ID = "adaptive_rerank_instruction"
RERANK_INSTRUCTION_KEY = "rerank_instruction"


def teacher_events_from_wm(
    wm: Mapping[str, Any],
    *,
    turn_id: int = 0,
    query: str | None = None,
    allow_synthetic_heuristic: bool = False,
) -> list[HarnessEvent]:
    q = str(query if query is not None else wm.get("query") or "")
    instruction = str(wm.get(RERANK_INSTRUCTION_KEY) or "")
    if not instruction.strip():
        return skip_teacher_events(
            COMPONENT_ID,
            turn_id=turn_id,
            reason="no_rerank_instruction",
            teacher_kind=SOURCE_SKIP_UNTRIGGERED,
        )
    transform = obs_transform(
        COMPONENT_ID,
        turn_id=turn_id,
        observation={RERANK_INSTRUCTION_KEY: instruction},
        visible_to_student=False,
        metadata={
            "owner": "teacher_full",
            "student_must_not_see": True,
            "student_realizable": False,
            "source_type": SOURCE_SYNTHETIC,
            "teacher_kind": SOURCE_SYNTHETIC,
        },
    )
    if not allow_synthetic_heuristic:
        return skip_teacher_events(
            COMPONENT_ID,
            turn_id=turn_id,
            reason="no_capability_derived_action",
            teacher_kind=SOURCE_SKIP_UNTRIGGERED,
        )
    events = tag_source(
        [
            transform,
            model_action(
                "search_corpus",
                {"query": q},
                turn_id=turn_id,
                component_id=COMPONENT_ID,
                metadata={"student_realizable": True, "teacher_result_ids": [], "synthetic_heuristic": True},
            ),
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
