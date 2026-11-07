"""Environment NAMED validators (guide §7 / PR-08).

The generic realizability checks (``opd.realizability``) have three concrete
holes the guide calls out:

1. ``check_syntax`` only tests required-key presence, never argument TYPES;
2. tools without a registered precondition default to executable=True;
3. AppWorld's ``python_exec`` ``code`` argument is a free string the generic
   fact scan never inspects.

This package adds per-environment validators that return a THREE-VALUE
verdict (valid | invalid | uncertain) and are registered under
``(environment_name, environment_version, tool_schema_digest)`` — different
environments with same-named tools never share rules.  Any unknown
environment/action yields ``uncertain``; the formal OPD path treats that as
skip.  Environment checks may only read PUBLIC student-visible state — never
evaluator answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.observation import ToolSpec


VERDICT_VALID = "valid"
VERDICT_INVALID = "invalid"
VERDICT_UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class TargetValidation:
    verdict: str  # valid | invalid | uncertain
    reason_codes: tuple[str, ...]
    state_matches: bool
    syntax_valid: bool
    executable: bool | None  # None = unknown (uncertain)
    information_supported: bool | None  # None = unknown (uncertain)
    supporting_public_event_ids: tuple[str, ...] = ()
    acquire_first_action: Action | None = None

    @property
    def acceptable_for_opd(self) -> bool:
        return self.verdict == VERDICT_VALID


# --- registry ---------------------------------------------------------------

_VALIDATORS: dict[tuple[str, str, str], Any] = {}


def register_validator(environment_name: str, environment_version: str, tool_schema_digest: str, fn) -> None:
    _VALIDATORS[(environment_name, environment_version, tool_schema_digest)] = fn


def validator_for(environment_identity: Mapping[str, Any]):
    """Resolve the registered validator; unknown environments get a validator
    that returns ``uncertain`` (fail-closed for formal OPD)."""
    key = (
        str(environment_identity.get("name", "")),
        str(environment_identity.get("version", "")),
        str(environment_identity.get("tool_schema_hash", "")),
    )
    fn = _VALIDATORS.get(key)
    if fn is not None:
        return fn

    def _unknown(*args, **kwargs):
        return TargetValidation(
            verdict=VERDICT_UNCERTAIN,
            reason_codes=("unknown_environment_or_tool_schema",),
            state_matches=False,
            syntax_valid=False,
            executable=None,
            information_supported=None,
        )

    return _unknown


# --- shared syntax check (full JSON-Schema-style) -----------------------------

_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def check_syntax_full(action: ToolCall, catalog: Sequence[ToolSpec]) -> tuple[bool, tuple[str, ...]]:
    """Type-checked schema validation (guide §7.1: key presence is not enough)."""
    spec = next((t for t in catalog if t.name == action.name), None)
    if spec is None:
        return False, (f"unknown tool {action.name!r}",)
    schema = dict(spec.parameters_schema or {})
    props = dict(schema.get("properties", {}) or {})
    reasons: list[str] = []
    for r in [str(x) for x in schema.get("required", [])]:
        if r not in action.arguments:
            reasons.append(f"missing required argument {r!r}")
    for k, v in action.arguments.items():
        if props and k not in props:
            reasons.append(f"unexpected argument {k!r}")
            continue
        expected = str((props.get(k) or {}).get("type", ""))
        py_t = _JSON_TYPES.get(expected)
        if py_t is not None and v is not None:
            # bool is a subclass of int — keep them distinct
            if expected in ("integer", "number") and isinstance(v, bool):
                reasons.append(f"argument {k!r} must be {expected}, got bool")
            elif not isinstance(v, py_t):
                reasons.append(f"argument {k!r} must be {expected}, got {type(v).__name__}")
    return (not reasons), tuple(reasons)
