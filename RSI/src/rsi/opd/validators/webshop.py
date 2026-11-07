"""WebShop environment validator (guide §7.3)."""

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


def validate_webshop_target(
    *,
    action: Action,
    catalog: Sequence[ToolSpec],
    visible_state: Mapping[str, Any],
    public_event_ids: Sequence[str] = (),
) -> TargetValidation:
    """Page-type-aware legality.

    ``visible_state`` exposes the CURRENT page's real clickable set:
    ``page_type`` (search | results | item | purchase_done), ``clickables``
    (exact button/link ids or titles), and ``required_options_satisfied``.
    The pre-fix rule accepted any action whose text contained a product
    title string — buying from the wrong page or clicking a non-clickable
    title both passed.
    """
    if isinstance(action, Finish):
        return TargetValidation(
            verdict=VERDICT_VALID, reason_codes=(), state_matches=True,
            syntax_valid=True, executable=True, information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    assert isinstance(action, ToolCall)
    if action.name != "click":
        # search stays in the basic set; other tools are uncertain
        if action.name == "search":
            syntax_ok, reasons = check_syntax_full(action, catalog)
            return TargetValidation(
                verdict=VERDICT_VALID if syntax_ok else VERDICT_INVALID,
                reason_codes=reasons,
                state_matches=True,
                syntax_valid=syntax_ok,
                executable=True if syntax_ok else False,
                information_supported=True,
                supporting_public_event_ids=tuple(public_event_ids),
            )
        return TargetValidation(
            verdict=VERDICT_UNCERTAIN,
            reason_codes=(f"non_basic_tool:{action.name}",),
            state_matches=False, syntax_valid=False,
            executable=None, information_supported=None,
        )
    syntax_ok, reasons = check_syntax_full(action, catalog)
    if not syntax_ok:
        return TargetValidation(
            verdict=VERDICT_INVALID, reason_codes=reasons, state_matches=False,
            syntax_valid=False, executable=False, information_supported=None,
        )
    page_type = str(visible_state.get("page_type", ""))
    clickables = [str(c) for c in (visible_state.get("clickables") or [])]
    target = str(action.arguments.get("target", ""))
    if page_type == "item" and target.lower() == "buy now":
        options_ok = bool(visible_state.get("required_options_satisfied", False))
        return TargetValidation(
            verdict=VERDICT_VALID if options_ok else VERDICT_INVALID,
            reason_codes=() if options_ok else ("buy_now_with_unsatisfied_required_options",),
            state_matches=options_ok,
            syntax_valid=True,
            executable=options_ok,
            information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    if target in clickables or target.lower() in {c.lower() for c in clickables}:
        return TargetValidation(
            verdict=VERDICT_VALID, reason_codes=(), state_matches=True,
            syntax_valid=True, executable=True, information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    if not clickables:
        return TargetValidation(
            verdict=VERDICT_UNCERTAIN,
            reason_codes=("no_clickables_in_visible_state",),
            state_matches=False, syntax_valid=True,
            executable=None, information_supported=None,
        )
    return TargetValidation(
        verdict=VERDICT_INVALID,
        reason_codes=("target_not_clickable_on_current_page",),
        state_matches=False, syntax_valid=True,
        executable=False, information_supported=False,
    )
