"""Messages / tool-call conversion only (E04).

Does not execute tools, update working memory, retry format errors, or
decide episode end. Those stay on the original ``SlidingWindowSearchEnv``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

PROTOCOL_CHAT_COMPLETIONS_V1 = "chat_completions_v1"
# Nested ``function`` tools: OpenAI chat.completions / vLLM / Qwen.
CHAT_TOOLS_PROVIDER = "qwen_moonshot"


@dataclass
class ChatMessage:
    role: str
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.reasoning:
            payload["reasoning_content"] = self.reasoning
        return payload


@dataclass
class ParsedApiAction:
    """Normalized tool calls ready for original ActionBuilder / step_action."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    finish_reason: str | None = None
    parse_error: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.parse_error is None


def chat_tools_from_upstream_schemas(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep original parameter / enum / required fields. Nested function form."""
    tools: list[dict[str, Any]] = []
    for schema in schemas:
        if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
            tools.append(dict(schema))
            continue
        name = str(schema.get("name") or "")
        if not name:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description") or "",
                    "parameters": schema.get("parameters")
                    or {
                        "type": "object",
                        "properties": schema.get("properties") or {},
                        "required": list(schema.get("required") or []),
                    },
                },
            }
        )
    return tools


def messages_from_selected_context(
    *,
    system_prompt: str,
    wm_text: str | None,
    history: Sequence[Mapping[str, Any]],
    result_summaries: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """OpenAI chat messages from the original selected context window.

    ``history`` items are ``{"role": "assistant"|"tool"|"user", ...}`` already
    converted from original Action/Observation objects by the env bridge.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": system_prompt}]
    if wm_text:
        messages.append({"role": "user", "content": wm_text})
    n = len(history)
    for i, item in enumerate(history):
        messages.append(dict(item))
        if result_summaries and i < len(result_summaries) - 0:
            is_last = i == n - 1
            if (not is_last) and result_summaries[i]:
                messages.append({"role": "user", "content": result_summaries[i]})
    return messages


def parse_chat_completion(response: Mapping[str, Any]) -> ParsedApiAction:
    """Convert one chat.completions payload. Do not invent missing reasoning."""
    raw = dict(response)
    choices = raw.get("choices") or []
    if not choices:
        return ParsedApiAction(parse_error="No response choices received from API", raw_response=raw)
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    parsed = ParsedApiAction(
        reasoning=str(reasoning) if reasoning else None,
        finish_reason=None if finish is None else str(finish),
        raw_response=raw,
    )
    tool_calls = message.get("tool_calls") or []
    if finish == "tool_calls" or tool_calls:
        if not tool_calls:
            parsed.parse_error = "finish_reason=tool_calls but no tool_calls"
            return parsed
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            if not name:
                parsed.parse_error = "Tool call is missing function name"
                parsed.tool_calls = []
                return parsed
            args_raw = fn.get("arguments") or "{}"
            if isinstance(args_raw, Mapping):
                params = dict(args_raw)
            else:
                try:
                    params = json.loads(args_raw)
                except json.JSONDecodeError as exc:
                    parsed.parse_error = f"Invalid JSON arguments for tool {name}: {exc}"
                    parsed.tool_calls = []
                    return parsed
            parsed.tool_calls.append(
                {
                    "name": name,
                    "arguments": params,
                    "id": str(call.get("id") or "agent"),
                }
            )
        return parsed
    text = _message_text(message)
    if text:
        parsed.tool_calls.append(
            {"name": "user_text", "arguments": {"text": text}, "id": "agent"}
        )
        return parsed
    if finish in {None, "stop", "length"}:
        parsed.parse_error = "Reasoning-only action with no tool calls"
        return parsed
    parsed.parse_error = f"Unhandled finish_reason={finish!r} with empty content"
    return parsed


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def request_fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ChatCompletionsClient:
    """HTTP client. Retries resend the same JSON body. Never steps an env."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 3
    protocol: str = PROTOCOL_CHAT_COMPLETIONS_V1
    temperature: float = 1.0
    max_tokens: int = 2048

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "tools": [dict(t) for t in tools],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if extra:
            payload.update(dict(extra))
        body = json.dumps(payload).encode("utf-8")
        fingerprint = request_fingerprint(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                raw["_request_fingerprint"] = fingerprint
                raw["_protocol"] = self.protocol
                return raw
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"API request failed after {self.max_retries} identical retries: {last_error}")
