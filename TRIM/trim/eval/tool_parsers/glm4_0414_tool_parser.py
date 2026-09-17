"""vLLM tool-parser plugin for GLM-4-0414 native calls: ``name\\n{json}``.

GLM-4-32B-0414 writes function calls as assistant metadata + JSON body, which
vLLM surfaces as content like::

    search_corpus
    {"query": "smoke test"}

The stock ``glm45`` parser expects GLM-4.5/4.7 XML and drops that into
``content`` with no ``tool_calls``. Register with::

    --tool-parser-plugin <this file> --tool-call-parser glm4_0414
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager

_KNOWN_TOOL_NAMES = frozenset(
    {
        "search_corpus",
        "curate",
        "end_search",
        "grep_corpus",
        "read_document",
        "verify",
        "fan_out_search",
        "review_docs",
        "prune_chunks",
    }
)
_TOOL_NAMES_PATTERN = (
    "search_corpus|curate|end_search|grep_corpus|read_document|"
    "verify|fan_out_search|review_docs|prune_chunks"
)
_NAME_JSON_HEAD_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s`(\[{]))(" + _TOOL_NAMES_PATTERN + r")(?=\s*\n\s*\{)",
    re.IGNORECASE,
)
_ASSISTANT_SPLIT_RE = re.compile(r"<\|assistant\|>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_FENCE_BLOCK_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[^\n`]*\n?")


def _unwrap_markdown_fences(text: str) -> str:
    raw = str(text or "")
    prev = None
    while prev != raw:
        prev = raw
        raw = _FENCE_BLOCK_RE.sub(lambda m: "\n" + m.group(1).strip() + "\n", raw)
    return _OPEN_FENCE_RE.sub("\n", raw).strip()


def _extract_balanced_json(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str: str | None = None
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in {'"', "'"}:
            in_str = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _chunks(model_output: str) -> list[str]:
    text = str(model_output or "").strip()
    if not text:
        return []
    parts = _ASSISTANT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def extract_glm4_0414_calls(model_output: str) -> list[tuple[str, str]]:
    """Return ``(name, arguments_json)`` pairs, or empty if not this format."""
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in _chunks(model_output):
        cleaned = _unwrap_markdown_fences(_SPECIAL_TOKEN_RE.sub("", chunk).strip())
        for match in _NAME_JSON_HEAD_RE.finditer(cleaned):
            name = match.group(1)
            if name not in _KNOWN_TOOL_NAMES:
                continue
            rest = cleaned[match.end() :].lstrip()
            blob = _extract_balanced_json(rest, 0) if rest.startswith("{") else None
            if not blob:
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            args = json.dumps(parsed, ensure_ascii=False)
            key = (name, args)
            if key in seen:
                continue
            seen.add(key)
            found.append((name, args))
    return found


class Glm40414ToolParser(ToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._streamed_complete = False

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        del request
        calls = extract_glm4_0414_calls(model_output)
        if not calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        tool_calls = [
            ToolCall(
                type="function",
                function=FunctionCall(name=name, arguments=args),
            )
            for name, args in calls
        ]
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=None
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids
        if self._streamed_complete:
            return None
        current = self.extract_tool_calls(current_text, request)
        if not current.tools_called:
            return None
        previous = self.extract_tool_calls(previous_text, request)
        if previous.tools_called:
            return None
        self._streamed_complete = True
        deltas = [
            DeltaToolCall(
                id=call.id,
                type="function",
                index=index,
                function=DeltaFunctionCall(
                    name=call.function.name,
                    arguments=call.function.arguments,
                ),
            )
            for index, call in enumerate(current.tool_calls)
        ]
        return DeltaMessage(tool_calls=deltas)


ToolParserManager.register_module(name="glm4_0414", module=Glm40414ToolParser)
