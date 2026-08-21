from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

LEGAL_TOOLS = {"fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search"}


@dataclass(frozen=True)
class ToolSpanAudit:
    parsable: bool
    tool_name: str | None
    n_tool_name: int
    n_argument_key: int
    n_argument_value: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsable": self.parsable,
            "tool_name": self.tool_name,
            "n_tool_name": self.n_tool_name,
            "n_argument_key": self.n_argument_key,
            "n_argument_value": self.n_argument_value,
            "error": self.error,
        }


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    m = re.search(r"(?:^|\n)to=([A-Za-z_][A-Za-z0-9_]*)", text)
    if not m:
        raise ValueError("missing to=<tool> marker")
    name = m.group(1)
    if name not in LEGAL_TOOLS:
        raise ValueError(f"invalid tool: {name}")
    rest = text[m.end():]
    jm = re.search(r"\{.*\}", rest, flags=re.DOTALL)
    args = json.loads(jm.group(0)) if jm else {}
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return name, args


def audit_tool_call_span(text: str) -> ToolSpanAudit:
    try:
        name, args = parse_tool_call(text)
        key_count = len(args)
        val_count = sum(1 if not isinstance(v, list) else len(v) for v in args.values())
        return ToolSpanAudit(True, name, 1, key_count, val_count)
    except Exception as exc:
        return ToolSpanAudit(False, None, 0, 0, 0, str(exc))


def require_parsable_tool_calls(texts: list[str]) -> dict[str, Any]:
    audits = [audit_tool_call_span(t) for t in texts]
    ok = [a for a in audits if a.parsable]
    rate = len(ok) / max(1, len(audits))
    if texts and rate < 1.0:
        raise AssertionError("tool span parsable rate must be 100%")
    return {"n": len(texts), "parsable_rate": rate, "audits": [a.to_dict() for a in audits]}
