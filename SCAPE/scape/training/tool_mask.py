"""Tool-call span masks for tool-token OPD.

Identifies token spans for:
- tool name
- argument keys
- argument values
- end-of-tool / end_search markers
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TOOL_NAME_RE = re.compile(
    r"(?P<prefix>(?:to=|name\s*=\s*|tool\s*=\s*|call\s+)?)"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
END_SEARCH_MARKERS = ("end_search", "</tool_call>", "<|call|>", "to=end_search")


@dataclass(frozen=True)
class TokenSpan:
    kind: str  # tool_name | argument_key | argument_value | end_search | other
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


def _find_all(pattern: re.Pattern[str], text: str) -> list[re.Match[str]]:
    return list(pattern.finditer(text))


def extract_tool_name_spans(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for m in _find_all(TOOL_NAME_RE, text):
        # Prefer explicit call forms; skip bare words inside JSON values by
        # requiring a tool-ish prefix OR being on a line that looks like a call.
        prefix = m.group("prefix") or ""
        name = m.group("name")
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : text.find("\n", m.start())]
        looks_like_call = bool(prefix.strip()) or ("to=" in line) or line.strip().startswith("call ")
        if not looks_like_call:
            continue
        name_start = m.start("name")
        name_end = m.end("name")
        spans.append(TokenSpan("tool_name", name_start, name_end, name))
    return spans


def extract_argument_spans(text: str) -> tuple[list[TokenSpan], list[TokenSpan]]:
    """Extract argument key/value spans from embedded JSON objects."""
    key_spans: list[TokenSpan] = []
    val_spans: list[TokenSpan] = []
    # Find JSON-like objects
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        blob = m.group(0)
        base = m.start()
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # Fallback: regex "key": value
            for km in re.finditer(r'"([^"]+)"\s*:', blob):
                key_spans.append(
                    TokenSpan(
                        "argument_key",
                        base + km.start(1),
                        base + km.end(1),
                        km.group(1),
                    )
                )
            for vm in re.finditer(r':\s*"([^"]*)"', blob):
                val_spans.append(
                    TokenSpan(
                        "argument_value",
                        base + vm.start(1),
                        base + vm.end(1),
                        vm.group(1),
                    )
                )
            continue
        if not isinstance(obj, dict):
            continue
        # Locate keys/values by scanning the blob text
        for key, value in obj.items():
            key_pat = f'"{key}"'
            kpos = blob.find(key_pat)
            if kpos >= 0:
                # span over key without quotes for token alignment flexibility
                key_spans.append(
                    TokenSpan("argument_key", base + kpos + 1, base + kpos + 1 + len(key), key)
                )
            if isinstance(value, str):
                vpat = json.dumps(value)
                vpos = blob.find(vpat, kpos if kpos >= 0 else 0)
                if vpos >= 0:
                    # inside quotes
                    val_spans.append(
                        TokenSpan(
                            "argument_value",
                            base + vpos + 1,
                            base + vpos + len(vpat) - 1,
                            value,
                        )
                    )
            else:
                vpat = json.dumps(value)
                vpos = blob.find(vpat, kpos if kpos >= 0 else 0)
                if vpos >= 0:
                    val_spans.append(
                        TokenSpan(
                            "argument_value",
                            base + vpos,
                            base + vpos + len(vpat),
                            vpat,
                        )
                    )
    return key_spans, val_spans


def extract_end_search_spans(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for marker in END_SEARCH_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            spans.append(TokenSpan("end_search", idx, idx + len(marker), marker))
            start = idx + len(marker)
    return spans


def build_tool_token_mask(
    text: str,
    *,
    include_name: bool = True,
    include_arg_keys: bool = True,
    include_arg_values: bool = True,
    include_end_search: bool = True,
) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    if include_name:
        spans.extend(extract_tool_name_spans(text))
    if include_arg_keys or include_arg_values:
        keys, vals = extract_argument_spans(text)
        if include_arg_keys:
            spans.extend(keys)
        if include_arg_values:
            spans.extend(vals)
    if include_end_search:
        spans.extend(extract_end_search_spans(text))
    spans.sort(key=lambda s: (s.start, s.end, s.kind))
    return spans


def spans_to_char_mask(text: str, spans: Sequence[TokenSpan]) -> list[bool]:
    mask = [False] * len(text)
    for sp in spans:
        for i in range(max(0, sp.start), min(len(text), sp.end)):
            mask[i] = True
    return mask


def align_char_mask_to_tokens(
    text: str,
    char_mask: Sequence[bool],
    token_offsets: Sequence[tuple[int, int]],
) -> list[bool]:
    """Map char-level mask onto token offsets [(start, end), ...]."""
    out: list[bool] = []
    for start, end in token_offsets:
        if start >= end or start >= len(char_mask):
            out.append(False)
            continue
        end = min(end, len(char_mask))
        out.append(any(char_mask[start:end]))
    return out


def tool_loss_mask_from_response(
    response_text: str,
    token_offsets: Sequence[tuple[int, int]] | None = None,
    **span_kwargs: Any,
) -> dict[str, Any]:
    spans = build_tool_token_mask(response_text, **span_kwargs)
    char_mask = spans_to_char_mask(response_text, spans)
    token_mask = None
    if token_offsets is not None:
        token_mask = align_char_mask_to_tokens(response_text, char_mask, token_offsets)
    return {
        "spans": [s.to_dict() for s in spans],
        "char_mask": char_mask,
        "token_mask": token_mask,
        "n_tool_name": sum(1 for s in spans if s.kind == "tool_name"),
        "n_argument_key": sum(1 for s in spans if s.kind == "argument_key"),
        "n_argument_value": sum(1 for s in spans if s.kind == "argument_value"),
        "n_end_search": sum(1 for s in spans if s.kind == "end_search"),
    }


def legal_tool_names(extra: Iterable[str] | None = None) -> list[str]:
    base = [
        "search",
        "grep",
        "read_document",
        "curate",
        "verify",
        "end_search",
        "multi_tool_use",
    ]
    if extra:
        base.extend(list(extra))
    # stable unique
    seen: set[str] = set()
    out: list[str] = []
    for n in base:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
