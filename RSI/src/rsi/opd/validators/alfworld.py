"""ALFWorld environment validator (guide §7.2)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.observation import ToolSpec
from rsi.opd.validators.base import (
    VERDICT_INVALID,
    VERDICT_UNCERTAIN,
    VERDICT_VALID,
    TargetValidation,
    check_syntax_full,
)


def validate_alfworld_target(
    *,
    action: Action,
    catalog: Sequence[ToolSpec],
    visible_state: Mapping[str, Any],
    public_event_ids: Sequence[str] = (),
) -> TargetValidation:
    """Legality against the CURRENT admissible commands.

    ``visible_state`` must expose ``admissible_commands`` taken from the
    anchor's Observation.structured — the commands legal at the decision
    point, NOT the commands after the teacher's own exploration.  A command
    that is legal but not in the visible set is INVALID (the pre-fix behavior
    accepted any string matching a command-shaped pattern).
    """
    if isinstance(action, Finish):
        return TargetValidation(
            verdict=VERDICT_VALID,
            reason_codes=(),
            state_matches=True,
            syntax_valid=True,
            executable=True,
            information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    assert isinstance(action, ToolCall)
    if action.name != "command":
        return TargetValidation(
            verdict=VERDICT_UNCERTAIN,
            reason_codes=(f"non_basic_tool:{action.name}",),
            state_matches=False,
            syntax_valid=False,
            executable=None,
            information_supported=None,
        )
    syntax_ok, reasons = check_syntax_full(action, catalog)
    if not syntax_ok:
        return TargetValidation(
            verdict=VERDICT_INVALID,
            reason_codes=reasons,
            state_matches=False,
            syntax_valid=False,
            executable=False,
            information_supported=None,
        )
    text = str(action.arguments.get("text", "")).strip()
    admissible = [str(c) for c in (visible_state.get("admissible_commands") or [])]
    if not admissible:
        return TargetValidation(
            verdict=VERDICT_UNCERTAIN,
            reason_codes=("no_admissible_commands_in_visible_state",),
            state_matches=False,
            syntax_valid=True,
            executable=None,
            information_supported=None,
        )
    if text in admissible:
        return TargetValidation(
            verdict=VERDICT_VALID,
            reason_codes=(),
            state_matches=True,
            syntax_valid=True,
            executable=True,
            information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    # case-insensitive exact match as a fallback (engine is case-sensitive;
    # a normalization here must preserve the original string in the row)
    lowered = {c.lower(): c for c in admissible}
    if text.lower() in lowered:
        return TargetValidation(
            verdict=VERDICT_VALID,
            reason_codes=("normalized_case_match",),
            state_matches=True,
            syntax_valid=True,
            executable=True,
            information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    return TargetValidation(
        verdict=VERDICT_INVALID,
        reason_codes=("command_not_in_current_admissible_set",),
        state_matches=False,
        syntax_valid=True,
        executable=False,
        information_supported=False,
    )


register_alfworld = None  # registration happens via opd.validators.__init__
