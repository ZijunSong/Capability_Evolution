"""Token alignment between student and teacher views."""

from __future__ import annotations

from dataclasses import dataclass

from training.opd._policy_backend import OPDTransition


@dataclass
class AlignedTokens:
    student_prefix_len: int
    teacher_prefix_len: int
    action_start_student: int
    action_start_teacher: int
    action_ids: list[int]
    action_mask: list[bool]


def align_action_tokens(transition: OPDTransition) -> AlignedTokens:
    student_len = len(transition.student_input_ids)
    teacher_len = len(transition.teacher_input_ids)
    action_len = len(transition.action_ids)
    return AlignedTokens(
        student_prefix_len=student_len,
        teacher_prefix_len=teacher_len,
        action_start_student=max(0, student_len - action_len),
        action_start_teacher=max(0, teacher_len - action_len),
        action_ids=list(transition.action_ids),
        action_mask=list(transition.action_mask),
    )


def is_critical_action_token(token_text: str) -> bool:
    critical = {
        "search_corpus",
        "grep_corpus",
        "read_document",
        "curate",
        "verify",
        "end_search",
        "query",
    }
    return any(c in token_text for c in critical)
