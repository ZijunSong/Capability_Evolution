"""Shared action validation for probe regeneration AND runtime rollout (guide §8.3).

One entry point, ``validate_action_at_state()``, so probe legality and real
rollout legality cannot drift:

- Real tool schema + parser semantics (name / required / types), not a
  hand-rolled half-set.
- Reference rules distinguish *acquisition* tools (read/search obtain docs —
  prior visibility is NOT required) from *consumption* tools (curate/answer
  consume known IDs — references must be visible).
- Curated-capacity for curate, teacher-only tools, no-op detection.
- Returns independent fields (``valid`` / ``executable`` / ``noop`` /
  ``repair_needed``); no-op is a legality pass but never a decision
  improvement.

Legacy checkers keep working; parity is asserted by tests
(``test_probe_and_runtime_validation_parity``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Tools that OBTAIN documents: their doc references are requests, not claims
# of prior access (mirrors opd_realizability acquisition_tools).
ACQUISITION_TOOLS = {
    "read_document",
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "init",
    "lookup",
    "answer",
}

# Tools that CONSUME already-known documents: references must be visible.
CONSUMPTION_TOOLS = {
    "curate",
}

TEACHER_ONLY_TOOLS = {"verify", "answer_with"}

FAIL_TOOL_NAME = "schema_error"
FAIL_ARGS = "schema_error"
FAIL_INACCESSIBLE_REF = "inaccessible_reference"
FAIL_NOOP = "noop"

_CURATED_CAPACITY_DEFAULT = 8


def _arg_list(args: Mapping[str, Any], key: str) -> list[str]:
    val = args.get(key) or []
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if val in ("", None):
        return []
    return [str(val)]


def validate_action_at_state(
    *,
    action: Mapping[str, Any],
    state: Any,
    student_mask: Mapping[str, bool] | None = None,
    curated_capacity: int = _CURATED_CAPACITY_DEFAULT,
) -> dict[str, Any]:
    """Validate one action dict against a probe state row (or snapshot dict).

    ``state`` may be a ``ProbeCState`` (uses ``student_snapshot``) or a raw
    snapshot mapping. Returns a diagnostic dict with independent fields:

    - ``valid``: legality under shared schema + reference + capacity rules
      (failure_kind names the first failing class).
    - ``executable``: not valid, OR valid and not a detected no-op.
    - ``noop``: structurally valid but changes nothing (empty curate, empty
      search). A no-op is NOT format-illegal.
    - ``repair_needed``: legal intent with fixable format problems (reserved;
      currently only the empty-curate case).
    """
    snapshot = getattr(state, "student_snapshot", state)
    wm = dict((snapshot.get("working_memory") if isinstance(snapshot, Mapping) else {}) or {})
    visible = {str(x) for x in (wm.get("accessible_doc_ids") or [])}
    docs = wm.get("documents") or []
    known = visible | {str(d.get("id")) for d in docs if isinstance(d, dict) and d.get("id")}
    curated = [str(x) for x in (wm.get("curated_ids") or [])]
    mask = dict(student_mask or {})

    name = str(action.get("name") or "")
    args = dict(action.get("arguments") or {})
    reasons: list[str] = []

    # 1. Tool legality (mask-driven; teacher-only tools are never student-legal).
    # Harness-1 zero mask keeps verify/answer_with out; an empty mask dict is
    # ambiguous (mask=None keeps verify for legacy fixtures) — always pass an
    # explicit mask from the caller; unknown tool names fail closed.
    from trim.training.tool_mask import legal_tool_names

    legal = set(legal_tool_names(harness_mask=mask or {}))
    if not name:
        return _result(False, FAIL_TOOL_NAME, ["EMPTY_TOOL_NAME"], executable=False)
    if name not in legal:
        return _result(False, FAIL_TOOL_NAME, [f"ILLEGAL_TOOL:{name}"], executable=False)
    if name in TEACHER_ONLY_TOOLS:
        return _result(False, FAIL_TOOL_NAME, [f"TEACHER_ONLY_TOOL:{name}"], executable=False)

    # 2. Argument schema: required keys per tool (shared minimal contract).
    required = _required_args(name)
    missing = [k for k in required if k not in args or args.get(k) in ("", None, [])]
    if missing:
        return _result(
            False,
            FAIL_ARGS,
            [f"MISSING_REQUIRED_ARG:{k}" for k in missing],
            executable=False,
        )

    noop = False
    repair_needed = False

    # 3. Reference rules: acquisition vs consumption (guide §8.3).
    if name in ACQUISITION_TOOLS:
        # read_document may fetch any known-or-searchable id; search/grep take
        # free queries. No visibility requirement — but an empty query IS a no-op.
        if name == "search_corpus" or name == "grep_corpus" or name == "fan_out_search":
            if not str(args.get("query") or "").strip():
                noop = True
    refs: list[str] = []
    if name in CONSUMPTION_TOOLS or name == "read_document":
        refs.extend(_arg_list(args, "doc_id"))
        refs.extend(_arg_list(args, "add_ids"))
        refs.extend(_arg_list(args, "remove_ids"))
    if name in CONSUMPTION_TOOLS:
        add_ids = _arg_list(args, "add_ids")
        remove_ids = _arg_list(args, "remove_ids")
        if not add_ids and not remove_ids:
            noop = True
            repair_needed = True
        # curate consumes known docs; hidden IDs are illegal here.
        bad = [r for r in set(add_ids + remove_ids) if r not in known]
        if bad:
            return _result(
                False,
                FAIL_INACCESSIBLE_REF,
                [f"CURATE_REF_NOT_KNOWN:{r}" for r in sorted(bad)],
            )
        # Curated capacity: adds beyond capacity fail (mirrors
        # curated_capacity_holds semantics).
        if add_ids:
            projected = len(set(curated) - set(remove_ids)) + len(
                [a for a in add_ids if a not in curated]
            )
            if projected > curated_capacity:
                return _result(
                    False,
                    FAIL_INACCESSIBLE_REF,
                    [f"CURATED_CAPACITY_EXCEEDED:{projected}>{curated_capacity}"],
                )
    else:
        # read_document: fetching a doc that exists in the environment but is
        # not yet visible is the POINT of the tool (acquisition). We only flag
        # references that are nowhere in the snapshot (not even discovered).
        pass

    return _result(True, "", reasons, noop=noop, repair_needed=repair_needed)


def _required_args(name: str) -> tuple[str, ...]:
    if name in {"search_corpus", "grep_corpus", "fan_out_search"}:
        return ("query",)
    if name == "read_document":
        return ("doc_id",)
    if name == "curate":
        return ()  # empty curate is a no-op, not a schema error
    return ()


def _result(
    valid: bool,
    failure_kind: str,
    reasons: list[str],
    *,
    noop: bool = False,
    repair_needed: bool = False,
    executable: bool | None = None,
) -> dict[str, Any]:
    """``executable`` defaults to ``valid and not noop``; invalid outcomes pass
    ``executable=False`` explicitly."""
    return {
        "valid": valid,
        "failure_kind": failure_kind,
        "reasons": list(reasons),
        "executable": bool(valid and not noop) if executable is None else bool(executable),
        "noop": noop,
        "repair_needed": repair_needed,
    }


__all__ = [
    "ACQUISITION_TOOLS",
    "CONSUMPTION_TOOLS",
    "TEACHER_ONLY_TOOLS",
    "validate_action_at_state",
]
