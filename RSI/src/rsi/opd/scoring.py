"""Teacher-forced scoring of aligned targets (task book §8.8).

Teacher and Student score the *same target tokens* under their own prefixes.
The Teacher prefix is its private context at target-generation time — it must
not contain the already-generated target answer, future tool outputs, or
guidance text that copies the target into the prompt.

Recipe-projected basic actions are generally NOT the Teacher's original
sampled action; their scores are labeled ``teacher_scored_projected_target``
and must never be reported as the sampling probability of the original
advanced action.

If the Teacher's catalog only exposed advanced tools, the scoring view must
consistently expose the basic action schema under a frozen, versioned rendering
rule (``TEACHER_SCORING_VIEW_VERSION``).  A missing legal Teacher scoring view
means skip — never fall back to the first-round prompt, an empty prompt or the
Student's identical prompt to manufacture OPD data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rsi.contracts.action import encode_action, Action
from rsi.contracts.events import Event
from rsi.contracts.observation import ToolSpec
from rsi.contracts.task import PublicTask
from rsi.contracts.trace import AlignmentResult

TEACHER_SCORING_VIEW_VERSION = "teacher_scoring_view_v1"


@dataclass(frozen=True)
class TargetScore:
    target_token_ids: tuple[int, ...]
    teacher_logprobs: tuple[float, ...]
    student_logprobs: tuple[float, ...]
    teacher_prompt_token_ids: tuple[int, ...]
    student_prompt_token_ids: tuple[int, ...]
    target_kind: str  # native | teacher_scored_projected_target
    scoring_view_version: str = TEACHER_SCORING_VIEW_VERSION


def render_teacher_scoring_view(
    *,
    task: PublicTask,
    catalog: tuple[ToolSpec, ...],
    teacher_prefix_events: Sequence[Event],
    target_action: Action,
) -> str:
    """Frozen rendering rule for the Teacher scoring view.

    Exposes the *basic* action schema (whatever the catalog is) plus the
    Teacher's private prefix; the target action itself is NOT included in the
    prompt (it is scored, not shown).
    """
    from rsi.runtime.context import render_header, render_event

    header = render_header(task, catalog)
    body = "\n".join(render_event(e) for e in teacher_prefix_events)
    return (
        f"/* {TEACHER_SCORING_VIEW_VERSION} */\n"
        f"{header}\n\nTeacher private history:\n{body}\n\n"
        "Score the next basic action (not shown) under this context.\n"
    )


def score_aligned_target(
    *,
    policy: Any,
    teacher_policy: Any,
    task: PublicTask,
    catalog: tuple[ToolSpec, ToolSpec] | tuple[ToolSpec, ...],
    teacher_prefix_events: Sequence[Event],
    student_prompt_text: str,
    alignment: AlignmentResult,
    student_prompt_token_ids: Sequence[int] | None = None,
) -> TargetScore | None:
    """Teacher-force the target tokens under both contexts.

    Returns None when a legal Teacher scoring view does not exist (skip — no
    fallback prompts).
    """
    if alignment.target_action is None or alignment.route == "skip":
        return None
    if alignment.target_kind not in ("native", "teacher_scored_projected_target"):
        return None
    from rsi.providers.base import PolicyProvider  # noqa: F401  (documentation anchor)

    target_text = encode_action(alignment.target_action)
    # Student side: exact recorded prompt ids when provided (never re-render for
    # training); otherwise render once and keep the ids (test_exact_prompt_ids).
    if student_prompt_token_ids is not None:
        student_prompt_ids = list(student_prompt_token_ids)
    else:
        student_prompt_ids = policy.encode(student_prompt_text)
    target_ids = policy.encode(target_text, add_bos=False) if _accepts_add_bos(policy) else policy.encode(target_text)
    # Strip a BOS that encode() may prepend so target ids are pure action tokens.
    if target_ids and target_ids[0] == 256:
        target_ids = target_ids[1:]

    teacher_view = render_teacher_scoring_view(
        task=task,
        catalog=tuple(catalog),
        teacher_prefix_events=teacher_prefix_events,
        target_action=alignment.target_action,
    )
    if not teacher_view.strip():
        return None
    teacher_prompt_ids = teacher_policy.encode(teacher_view)

    student_lp = policy.teacher_force_logprobs(student_prompt_ids, target_ids)
    teacher_lp = teacher_policy.teacher_force_logprobs(teacher_prompt_ids, target_ids)
    if len(student_lp) != len(target_ids) or len(teacher_lp) != len(target_ids):
        return None
    return TargetScore(
        target_token_ids=tuple(target_ids),
        teacher_logprobs=tuple(teacher_lp),
        student_logprobs=tuple(student_lp),
        teacher_prompt_token_ids=tuple(teacher_prompt_ids),
        student_prompt_token_ids=tuple(student_prompt_ids),
        target_kind=alignment.target_kind,
    )


def _accepts_add_bos(policy: Any) -> bool:
    import inspect

    try:
        sig = inspect.signature(policy.encode)
    except (TypeError, ValueError):
        return False
    return "add_bos" in sig.parameters
