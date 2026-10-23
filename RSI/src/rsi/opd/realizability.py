"""Three-layer realizability of an aligned target action (task book §8.3).

1. syntax legal — action and args fit the current basic schema
2. environment executable — the Student state satisfies the precondition
3. information realizable — every external fact / object reference the action
   needs is already in the Student's visible input, or obtainable via a legal
   basic acquire action first

Skipping only means skipping *internal* events; gaining necessary environment
information is never skippable.  A better Teacher *inference* over public
inputs is legitimate transferable skill, not leakage; what is forbidden is
hidden labels, unpublic tool results and fork-only objects/state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue
from rsi.contracts.observation import ToolSpec
from rsi.contracts.trace import RealizabilityReport


@dataclass(frozen=True)
class SchemaCheckResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def check_syntax(action: Action, catalog: Sequence[ToolSpec]) -> SchemaCheckResult:
    if isinstance(action, Finish):
        return SchemaCheckResult(ok=True)
    assert isinstance(action, ToolCall)
    spec = next((t for t in catalog if t.name == action.name), None)
    if spec is None:
        return SchemaCheckResult(ok=False, reasons=(f"unknown tool {action.name!r}",))
    schema = dict(spec.parameters_schema or {})
    required = [str(r) for r in schema.get("required", [])]
    props = dict(schema.get("properties", {}) or {})
    reasons: list[str] = []
    for r in required:
        if r not in action.arguments:
            reasons.append(f"missing required argument {r!r}")
    for k in action.arguments:
        if props and k not in props:
            reasons.append(f"unexpected argument {k!r}")
    return SchemaCheckResult(ok=not reasons, reasons=tuple(reasons))


# --- precondition registry -------------------------------------------------
# Environment adapters register precondition predicates per tool.  Default: any
# syntactically legal call is executable.  This is an interface for real state
# preconditions (file exists, key present, ...), not a benchmark router.

PreconditionFn = Any  # (action: ToolCall, state_view: Mapping[str, Any]) -> tuple[bool, str | None]

_PRECONDITIONS: dict[str, PreconditionFn] = {}


def register_precondition(tool_name: str, fn: PreconditionFn) -> None:
    _PRECONDITIONS[tool_name] = fn


def check_precondition(action: Action, state_view: Mapping[str, Any]) -> tuple[bool, str | None]:
    if isinstance(action, Finish):
        return True, None
    fn = _PRECONDITIONS.get(action.name)
    if fn is None:
        return True, None
    return fn(action, state_view)


# --- information realizability --------------------------------------------

# Fact refs are opaque strings (paths, object ids, api refs) that the action
# *consumes*.  Write/put targets are outputs the action itself creates, not
# required prior facts.  Content/value arguments are scanned for fact-like
# tokens (a write whose content references a private object id is a leak).
CONSUMING_TOOLS = frozenset({"read_file", "get", "scan", "list_dir", "lookup", "query"})
CONSUMING_ARG_KEYS = ("path", "key", "id", "source", "file", "ref")


def extract_fact_refs(action: Action) -> tuple[str, ...]:
    """External facts the action depends on (consumed argument values)."""
    if isinstance(action, Finish):
        return ()
    refs: list[str] = []
    for k, v in action.arguments.items():
        if action.name in CONSUMING_TOOLS and k in CONSUMING_ARG_KEYS:
            if isinstance(v, str) and v:
                refs.append(v)
        elif k in ("content", "value", "text", "answer"):
            if isinstance(v, str) and v:
                refs.extend(_fact_tokens(v))
            elif isinstance(v, list):
                refs.extend(x for x in v if isinstance(x, str) and x)
    return tuple(refs)


def _fact_tokens(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for tok in text.replace("\n", " ").split(" "):
        tok = tok.strip(" ,;:()[]{}\"'")
        if _looks_like_fact_ref(tok) and len(tok) >= 2:
            out.append(tok)
    return tuple(out)


def check_information(
    action: Action,
    *,
    student_fact_refs: Sequence[str],
    teacher_private_fact_refs: Sequence[str] = (),
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Returns (realizable, missing_refs, leaked_refs).

    A needed ref is satisfied if it is in the student's visible refs.  Refs
    only present in teacher-private events are *leaks* unless re-acquirable.
    """
    needed = extract_fact_refs(action)
    student_refs = set(student_fact_refs)
    private_refs = set(teacher_private_fact_refs)
    missing = tuple(r for r in needed if r not in student_refs)
    leaked = tuple(r for r in missing if r in private_refs)
    # A ref that is not an obvious external fact (short generic strings) is
    # treated as constructible reasoning content, not an external fact.
    missing = tuple(r for r in missing if len(r) >= 2 and r not in private_refs and _looks_like_fact_ref(r))
    leaked = tuple(r for r in leaked if _looks_like_fact_ref(r))
    return (not missing and not leaked, missing, leaked)


def _looks_like_fact_ref(s: str) -> bool:
    return ("/" in s) or ("." in s) or s.startswith("id:") or s.startswith("ref:")


def build_realizability_report(
    action: Action,
    *,
    catalog: Sequence[ToolSpec],
    state_view: Mapping[str, Any],
    student_fact_refs: Sequence[str],
    teacher_private_fact_refs: Sequence[str] = (),
) -> RealizabilityReport:
    syn = check_syntax(action, catalog)
    pre_ok, pre_reason = check_precondition(action, state_view)
    info_ok, missing, leaked = check_information(
        action,
        student_fact_refs=student_fact_refs,
        teacher_private_fact_refs=teacher_private_fact_refs,
    )
    reasons = list(syn.reasons)
    if pre_reason:
        reasons.append(pre_reason)
    if leaked:
        reasons.append(f"teacher-private fact refs: {list(leaked)}")
    if missing:
        reasons.append(f"missing fact refs (acquire first): {list(missing)}")
    acquire = None
    if missing and not leaked and isinstance(action, ToolCall):
        acquire = _suggest_acquire_action(missing[0])
    return RealizabilityReport(
        syntax_legal=syn.ok,
        precondition_satisfied=pre_ok,
        information_realizable=info_ok,
        reasons=tuple(reasons),
        acquire_first_action=acquire,
    )


def _suggest_acquire_action(ref: str) -> Action | None:
    """A legal basic action that re-acquires the missing observation.

    If the fact can be re-read with a basic tool, the alignment may supervise
    *that* action first (acquire-then-supervise); otherwise the step is skipped.
    """
    if ref.endswith(".txt") or ref.endswith(".json") or "/" in ref:
        return ToolCall(name="read_file", arguments={"path": ref})
    if ref.startswith("id:"):
        return ToolCall(name="get", arguments={"key": ref})
    return None
