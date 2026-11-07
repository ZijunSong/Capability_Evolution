"""AppWorld environment validator (guide §7.4)."""

from __future__ import annotations

import ast
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


def _collect_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """App roots reachable from ``apis`` (e.g. ``spotify`` in
    ``apis.spotify.search(...)``) and loaded names."""
    api_calls: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node
            prev = None
            while isinstance(root, ast.Attribute):
                prev = root
                root = root.value
            if isinstance(root, ast.Name) and root.id == "apis" and prev is not None:
                # prev is the FIRST attribute after apis → the app name
                api_calls.add(prev.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    return api_calls, loaded


def validate_appworld_target(
    *,
    action: Action,
    catalog: Sequence[ToolSpec],
    visible_state: Mapping[str, Any],
    public_event_ids: Sequence[str] = (),
) -> TargetValidation:
    """AST-validated python_exec targets.

    ``visible_state`` exposes the decision-point session facts the STUDENT
    can see: ``session_variables`` (vars defined in the anchor's public
    python_exec history), ``visible_api_names`` (APIs shown in public docs
    output).  Static AST cannot prove arbitrary semantics — anything
    non-static (getattr/eval/dynamic names) returns ``uncertain`` and the
    formal path skips (guide §7.4).
    """
    if isinstance(action, Finish):
        return TargetValidation(
            verdict=VERDICT_VALID, reason_codes=(), state_matches=True,
            syntax_valid=True, executable=True, information_supported=True,
            supporting_public_event_ids=tuple(public_event_ids),
        )
    assert isinstance(action, ToolCall)
    if action.name != "python_exec":
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
    code = action.arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        return TargetValidation(
            verdict=VERDICT_INVALID,
            reason_codes=("code_not_a_string",),
            state_matches=False, syntax_valid=False,
            executable=False, information_supported=None,
        )
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return TargetValidation(
            verdict=VERDICT_INVALID,
            reason_codes=(f"python_syntax_error:{e.lineno}",),
            state_matches=False, syntax_valid=False,
            executable=False, information_supported=None,
        )
    # dynamic constructs cannot be validated statically → uncertain (skip)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
            "eval", "exec", "getattr", "globals", "locals", "__import__"
        ):
            return TargetValidation(
                verdict=VERDICT_UNCERTAIN,
                reason_codes=(f"dynamic_construct:{type(node).__name__}",),
                state_matches=False, syntax_valid=True,
                executable=None, information_supported=None,
            )
    api_calls, loaded = _collect_names(tree)
    session_vars = set(str(v) for v in (visible_state.get("session_variables") or []))
    # names loaded but never defined in the session and not apis/module-level
    session_missing = sorted(
        n for n in loaded
        if n not in session_vars and n not in ("apis", "requester", "json", "print")
        and not n.isupper()
    )
    if session_missing:
        return TargetValidation(
            verdict=VERDICT_INVALID,
            reason_codes=(f"session_variables_missing:{','.join(session_missing[:5])}",),
            state_matches=False, syntax_valid=True,
            executable=False, information_supported=False,
        )
    visible_apis = set(str(v) for v in (visible_state.get("visible_api_names") or []))
    # only flag API roots that the public history HAS shown something about
    if visible_apis and api_calls and not api_calls.issubset(visible_apis | {"api_docs", "supervisor"}):
        unknown = sorted(api_calls - visible_apis - {"api_docs", "supervisor"})
        return TargetValidation(
            verdict=VERDICT_UNCERTAIN,
            reason_codes=(f"api_root_not_in_visible_history:{','.join(unknown[:5])}",),
            state_matches=False, syntax_valid=True,
            executable=None, information_supported=None,
        )
    return TargetValidation(
        verdict=VERDICT_VALID, reason_codes=(), state_matches=True,
        syntax_valid=True, executable=True, information_supported=True,
        supporting_public_event_ids=tuple(public_event_ids),
    )
