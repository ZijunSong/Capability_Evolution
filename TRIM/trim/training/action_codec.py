"""Canonical Student-legal action render / parse codec.

Formal SR-OPD targets come from this codec, not free-form Teacher text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


ORDER_INSENSITIVE_KEYS = frozenset(
    {"add_ids", "remove_ids", "doc_ids", "evidence_ids", "result_ids", "sids"}
)
STUDENT_NATIVE_TOOLS = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "end_search",
)
HARNESS_G_STUDENT_NATIVE_TOOLS = (
    "init",
    "select",
    "lookup",
    "answer",
)
# Teacher-only; never a reduced-Student native tool.
TEACHER_ONLY_TOOLS = frozenset({"verify", "importance_tagging", "answer_with"})

_CALL_RE = re.compile(
    r"to=(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<body>\{.*?\})?",
    re.DOTALL,
)


def canonicalize_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(arguments or {})
    out: dict[str, Any] = {}
    for key in sorted(raw):
        value = raw[key]
        if key in ORDER_INSENSITIVE_KEYS and isinstance(value, list):
            out[key] = sorted(str(x) for x in value)
        else:
            out[key] = value
    return out


def canonicalize_action(action: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(action, "name") and hasattr(action, "arguments"):
        name = str(action.name)
        arguments = dict(action.arguments or {})
    else:
        name = str(action.get("name") or action.get("tool") or "")
        arguments = dict(action.get("arguments") or {})
    if not name:
        raise ValueError("action is missing a name")
    return {"name": name, "arguments": canonicalize_arguments(arguments)}


def render_action(action: Mapping[str, Any] | Any) -> str:
    canon = canonicalize_action(action)
    return (
        f"to={canon['name']}\n"
        f"{json.dumps(canon['arguments'], ensure_ascii=False)}\n"
    )


def format_tool_call_text(name: str, arguments: Mapping[str, Any]) -> str:
    """Backward-compatible alias used by same-state collection."""
    return render_action({"name": name, "arguments": arguments})


def parse_action(text: str) -> dict[str, Any]:
    blob = str(text or "").strip()
    if not blob:
        raise ValueError("empty action text")
    match = _CALL_RE.search(blob)
    if match:
        name = match.group("name")
        body = match.group("body") or "{}"
        try:
            arguments = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid action arguments: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("action arguments must be an object")
        return canonicalize_action({"name": name, "arguments": arguments})
    # Fallback: first token is the tool name, rest is JSON.
    first, _, rest = blob.partition("\n")
    name = first.replace("to=", "").strip()
    rest = rest.strip() or "{}"
    try:
        arguments = json.loads(rest)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid action arguments: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("action arguments must be an object")
    return canonicalize_action({"name": name, "arguments": arguments})


def validate_roundtrip(action: Mapping[str, Any] | Any) -> bool:
    canon = canonicalize_action(action)
    return parse_action(render_action(canon)) == canon
