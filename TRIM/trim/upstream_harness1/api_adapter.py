"""Messages / tool-call conversion only (E04).

Does not execute tools, update working memory, retry format errors, or
decide episode end. Those stay on the original ``SlidingWindowSearchEnv``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

PROTOCOL_CHAT_COMPLETIONS_V1 = "chat_completions_v1"
# Nested ``function`` tools: OpenAI chat.completions / vLLM / Qwen.
CHAT_TOOLS_PROVIDER = "qwen_moonshot"

_MAX_ERROR_BODY_CHARS = 8192
_TRANSPORT_RETRY_STATUS = frozenset({429, 502, 503, 504})
_CONFIG_ERROR_STATUS = frozenset({400, 401, 403, 404, 422})
_HARMONY_ERROR_MARKERS = (
    "harmonyerror",
    "harmony_error",
    "unexpected tokens remaining in message header",
    "could not decode header",
)


class ApiError(RuntimeError):
    """Base class for API-layer failures with structured metadata."""

    category: str = "api_error"

    def __init__(
        self,
        message: str,
        *,
        category: str | None = None,
        status: int | None = None,
        body: str | None = None,
        request_id: str | None = None,
        attempt: int | None = None,
        retry_events: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        self.status = status
        self.body = body
        self.request_id = request_id
        self.attempt = attempt
        self.retry_events = list(retry_events or [])

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "category": self.category,
            "message": str(self),
            "status": self.status,
            "body": self.body,
            "request_id": self.request_id,
            "attempt": self.attempt,
        }
        if self.retry_events:
            payload["retry_events"] = list(self.retry_events)
        return payload


class TransportError(ApiError):
    category = "transport_error"


class ConfigError(ApiError):
    category = "config_error"


class ServerParseError(ApiError):
    category = "model_output_parse_error"


class ServerErrorUnclassified(ApiError):
    category = "server_error_unclassified"


class QueryTimeoutError(TransportError):
    category = "query_timeout"


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
            # vLLM 0.25.1 normalizes reasoning_content → reasoning for GPT-OSS.
            payload["reasoning"] = self.reasoning
        return payload


@dataclass
class ParsedApiAction:
    """Normalized tool calls ready for original ActionBuilder / step_action."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    finish_reason: str | None = None
    api_finish_reason: str | None = None
    episode_finish_reason: str | None = None
    protocol_error: str | None = None
    parse_error: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.parse_error is None and self.protocol_error is None


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
    r"search_corpus|curate|end_search|grep_corpus|read_document|verify|fan_out_search|review_docs|prune_chunks"
)

_TOOL_CALL_IN_CONTENT_RE = re.compile(
    r"(?:<tool_call\b"
    r'|"name"\s*:\s*"(?:' + _TOOL_NAMES_PATTERN + r')"'
    r'|"(?:type|operation|function)"\s*:\s*"(?:' + _TOOL_NAMES_PATTERN + r')"'
    r"|<\|channel\|>|<\|call\|>"
    r"|(?:^|[\s{])(?:" + _TOOL_NAMES_PATTERN + r")\s*[:({]"
    r"|functions[\.\-\s](?:" + _TOOL_NAMES_PATTERN + r")"
    r"|\b(?:" + _TOOL_NAMES_PATTERN + r")\s*\(\s*\{"
    r")",
    re.IGNORECASE,
)

_TOOL_ARG_HINT_RE = re.compile(
    r'"(?:query|queries|doc_id|doc_ids|add_ids|remove_ids|pattern|claim|chunk_ids|importance)"\s*:',
    re.IGNORECASE,
)

_TOOL_ARG_ONLY_KEYS = frozenset(
    {
        "query",
        "queries",
        "doc_id",
        "doc_ids",
        "add_ids",
        "remove_ids",
        "pattern",
        "claim",
        "chunk_ids",
        "importance",
    }
)


def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if len(lines) < 2:
        return raw
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def _json_obj_looks_like_tool_call(obj: Mapping[str, Any], names: frozenset[str]) -> bool:
    if "name" in obj and isinstance(obj.get("name"), str):
        return str(obj["name"]) in names
    for key in ("type", "operation", "function"):
        val = obj.get(key)
        if isinstance(val, str) and val in names:
            return True
    recipient = obj.get("recipient")
    if isinstance(recipient, str) and "functions." in recipient:
        return True
    for key in obj:
        if key in names:
            return True
    if ("arguments" in obj or "parameters" in obj) and any(
        k in obj for k in ("name", "type", "function", "operation")
    ):
        return True
    keys = {str(k) for k in obj.keys()}
    if keys and keys <= _TOOL_ARG_ONLY_KEYS:
        return True
    return False


def _content_looks_like_tool_call(text: str, known_tool_names: frozenset[str] | None = None) -> bool:
    names = known_tool_names or _KNOWN_TOOL_NAMES
    raw = _strip_code_fences(str(text or "").strip())
    if not raw:
        return False
    if _TOOL_CALL_IN_CONTENT_RE.search(raw):
        return True
    if not raw.startswith("{"):
        return False
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        if any(f'"{name}"' in raw for name in names) and _TOOL_ARG_HINT_RE.search(raw):
            return True
        return False
    return isinstance(obj, dict) and _json_obj_looks_like_tool_call(obj, names)


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


def parse_chat_completion(response: Mapping[str, Any]) -> ParsedApiAction:
    """Convert one chat.completions payload. Do not invent missing reasoning."""
    raw = dict(response)
    choices = raw.get("choices") or []
    if not choices:
        return ParsedApiAction(parse_error="No response choices received from API", raw_response=raw)
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    api_finish = None if finish is None else str(finish)
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    parsed = ParsedApiAction(
        reasoning=str(reasoning) if reasoning else None,
        finish_reason=api_finish,
        api_finish_reason=api_finish,
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
            if "<|channel|>" in name or name.startswith("functions-"):
                parsed.protocol_error = f"Corrupted tool name: {name!r}"
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
            if not isinstance(params, dict):
                parsed.parse_error = f"Tool {name} arguments must be a JSON object"
                parsed.tool_calls = []
                return parsed
            parsed.tool_calls.append(
                {
                    "name": name,
                    "arguments": params,
                    "id": str(call.get("id") or "agent"),
                }
            )
        if any(str(c.get("name") or "") == "end_search" for c in parsed.tool_calls):
            parsed.episode_finish_reason = "explicit_end_search"
        return parsed

    text = _message_text(message)
    if text and _content_looks_like_tool_call(text):
        parsed.protocol_error = "tool_call_in_content"
        parsed.parse_error = "Response body contains tool-call syntax without structured tool_calls"
        return parsed

    if api_finish == "length":
        if text:
            parsed.protocol_error = "length_truncated"
            parsed.parse_error = "Response truncated (finish_reason=length) with partial content"
        else:
            parsed.protocol_error = "length_truncated"
            parsed.parse_error = "Reasoning-only action truncated (finish_reason=length)"
        return parsed

    if text:
        parsed.tool_calls.append(
            {"name": "user_text", "arguments": {"text": text}, "id": "agent"}
        )
        parsed.episode_finish_reason = "implicit_user_text"
        return parsed

    if finish in {None, "stop"}:
        parsed.parse_error = "Reasoning-only action with no tool calls"
        return parsed
    parsed.parse_error = f"Unhandled finish_reason={finish!r} with empty content"
    return parsed


def request_fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_error_body(exc: urllib.error.HTTPError) -> tuple[str, str | None]:
    body = ""
    request_id: str | None = None
    try:
        raw = exc.read(_MAX_ERROR_BODY_CHARS + 1)
        body = raw[:_MAX_ERROR_BODY_CHARS].decode("utf-8", errors="replace")
        if len(raw) > _MAX_ERROR_BODY_CHARS:
            body += "\n...(truncated)"
    except Exception:
        body = ""
    try:
        payload = json.loads(body)
        if isinstance(payload, Mapping):
            err = payload.get("error")
            if isinstance(err, Mapping):
                request_id = str(err.get("request_id") or err.get("requestId") or "") or None
            request_id = request_id or str(payload.get("request_id") or "") or None
    except json.JSONDecodeError:
        pass
    if request_id is None and exc.headers:
        request_id = exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id")
    return body, request_id


def _classify_http_error(status: int, body: str) -> type[ApiError]:
    lowered = body.lower()
    if status in _CONFIG_ERROR_STATUS:
        return ConfigError
    if status == 500 and any(marker in lowered for marker in _HARMONY_ERROR_MARKERS):
        return ServerParseError
    if status >= 500:
        return ServerErrorUnclassified
    return TransportError


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

    def build_payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        if tools:
            payload["tools"] = [dict(t) for t in tools]
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if extra:
            payload.update(dict(extra))
        return payload

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        payload = self.build_payload(
            messages,
            tools,
            extra=extra,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        body = json.dumps(payload).encode("utf-8")
        fingerprint = request_fingerprint(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        last_error: ApiError | None = None
        retry_events: list[dict[str, Any]] = []
        max_attempts = max(1, int(self.max_retries))
        for attempt in range(max_attempts):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    err = QueryTimeoutError(
                        "Query deadline exceeded before HTTP attempt",
                        attempt=attempt + 1,
                        retry_events=list(retry_events),
                    )
                    raise err
                timeout = min(timeout, remaining)
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                raw["_request_fingerprint"] = fingerprint
                raw["_protocol"] = self.protocol
                raw["_request_payload"] = payload
                raw["_transport_attempts"] = attempt + 1
                if retry_events:
                    raw["_transport_retry_events"] = list(retry_events)
                return raw
            except urllib.error.HTTPError as exc:
                err_body, request_id = _read_error_body(exc)
                err_cls = _classify_http_error(int(exc.code), err_body)
                last_error = err_cls(
                    f"HTTP {exc.code} from chat/completions: {err_body[:500] or exc.reason}",
                    status=int(exc.code),
                    body=err_body,
                    request_id=request_id,
                    attempt=attempt + 1,
                )
                retry_events.append(last_error.to_dict())
                if err_cls is ConfigError:
                    last_error.retry_events = list(retry_events)
                    raise last_error
                if int(exc.code) not in _TRANSPORT_RETRY_STATUS and int(exc.code) < 500:
                    last_error.retry_events = list(retry_events)
                    raise last_error
                if attempt + 1 >= max_attempts:
                    last_error.retry_events = list(retry_events)
                    raise last_error
                wait_s = min(2**attempt, 8)
                if deadline is not None:
                    wait_s = min(wait_s, max(0.0, deadline - time.monotonic()))
                if wait_s <= 0:
                    err = QueryTimeoutError(
                        "Query deadline exceeded during HTTP backoff",
                        attempt=attempt + 1,
                        retry_events=list(retry_events),
                    )
                    raise err
                time.sleep(wait_s)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = TransportError(
                    f"Transport failure on chat/completions attempt {attempt + 1}: {exc}",
                    attempt=attempt + 1,
                )
                retry_events.append(last_error.to_dict())
                if attempt + 1 >= max_attempts:
                    last_error.retry_events = list(retry_events)
                    raise last_error
                wait_s = min(2**attempt, 8)
                if deadline is not None:
                    wait_s = min(wait_s, max(0.0, deadline - time.monotonic()))
                if wait_s <= 0:
                    err = QueryTimeoutError(
                        "Query deadline exceeded during HTTP backoff",
                        attempt=attempt + 1,
                        retry_events=list(retry_events),
                    )
                    raise err
                time.sleep(wait_s)
        assert last_error is not None
        last_error.retry_events = list(retry_events)
        raise last_error
