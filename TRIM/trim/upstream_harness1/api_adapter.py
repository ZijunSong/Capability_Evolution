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
    r"(?:<\|tool_call\|>|<tool_call\b"
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


# --- Content-based tool-call extraction for models that emit tool calls in the
# response body instead of the OpenAI ``tool_calls`` field (e.g. GLM-4-32B-0414
# whose tokenizer has no ``<|tool_call|>`` token, so vLLM's glm45/glm47 parsers
# cannot extract structured tool_calls).  The model naturally emits several
# plain-text forms:
#   - ``func_name(arg_key=val, ...)``           (pythonic, optionally in a
#                                                ```python ... ``` fence)
#   - ``func_name\n{"arg": "val"}``             (name on its own line, JSON body)
#   - ``{"name": "func_name", "arguments": {..}} (plain JSON object)
# We attempt to extract the *last* recognizable call in the text so that
# trailing reasoning is not mistaken for content.
_GLM_FUNC_CALL_RE = re.compile(
    r"(?P<name>" + _TOOL_NAMES_PATTERN + r")\s*\((?P<args>.+?)\)",
    re.DOTALL,
)
# "func_name\n{json}" or "func_name: {json}" (with or without a trailing ``` fence)
_GLM_NAME_NEWLINE_JSON_RE = re.compile(
    r"(?P<name>" + _TOOL_NAMES_PATTERN + r")\s*[\n:]\s*(?P<json>\{.*?\})",
    re.DOTALL,
)
# "func_name: \"value\"" (colon-separated bare string)
_GLM_NAME_COLON_RE = re.compile(
    r"(?P<name>" + _TOOL_NAMES_PATTERN + r")\s*:\s*\"(?P<val>[^\"]*)\"",
    re.DOTALL,
)
_GLM_TOOLCALL_TAG_RE = re.compile(
    r"<\|tool_call\|>(?P<body>.*?)(?:<\|/tool_call\|>|</tool_call\|>|\Z)",
    re.DOTALL,
)
# GLM-4.7-style args: <|arg_key|>k<|/arg_key|><|arg_val|>v<|/arg_val|> or bare
# value text after <|/arg_key|> (when no <|arg_val|> wrapper is present).
_GLM_ARG_KV_RE = re.compile(
    r"<\|arg_key\|>(?P<key>.*?)(?:<\|/arg_key\|>|<\|/arg_val\|>)\s*(?:(?P<val>.*?)<\|/arg_val\|>)?",
    re.DOTALL,
)
_GLM_ARG_BARE_RE = re.compile(
    r"<\|arg_key\|>(?P<key>.+?)(?=\n<\|arg_key\|>|\Z)",
    re.DOTALL,
)


def _parse_glm_tool_call_args(raw_args: str, tool_name: str) -> dict[str, Any] | None:
    """Parse the argument portion of a content-embedded tool call.

    Handles pythonic ``key=val, key2=val2``, JSON objects, and bare string
    arguments.  Returns a dict or ``None`` on failure.
    """
    s = raw_args.strip()
    if not s:
        return {}
    # JSON object?
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Pythonic kwargs: key=value, key2=value2 (values may be quoted strings,
    # numbers, booleans, or JSON arrays/objects)
    if "=" in s:
        try:
            kwargs: dict[str, Any] = {}
            for part in _split_kwargs(s):
                if "=" not in part:
                    continue
                k, _, v = part.partition("=")
                k = k.strip()
                v = v.strip()
                # Try to parse value as JSON first (covers arrays/objects/nums)
                try:
                    v_parsed = json.loads(v)
                except json.JSONDecodeError:
                    v_parsed = v.strip("'\"")
                kwargs[k] = v_parsed
            if kwargs:
                return kwargs
        except Exception:
            pass
    # Bare positional arg (string)
    return {"query": s.strip("'\"")} if tool_name in ("search_corpus", "grep_corpus") else {"value": s.strip("'\"")}


def _split_kwargs(s: str) -> list[str]:
    """Split a pythonic argument list on top-level commas (respecting brackets/quotes)."""
    parts: list[str] = []
    depth = 0
    in_str: str | None = None
    start = 0
    for i, ch in enumerate(s):
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def extract_tool_calls_from_content(text: str, known_tool_names: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Extract structured tool calls from response content text.

    Tries several GLM-4-32B-0414 output patterns.  Returns the last
    recognizable call (the model often appends reasoning after the call).
    Each element is ``{"name": str, "arguments": dict, "id": "agent"}``.
    Returns an empty list if no tool call is found.
    """
    names = known_tool_names or _KNOWN_TOOL_NAMES
    raw_text = str(text or "")
    calls: list[dict[str, Any]] = []

    # 1. ``<|tool_call|>body<|/tool_call|>`` or ``<|tool_call|>body<|/arg_call|>``
    for m in _GLM_TOOLCALL_TAG_RE.finditer(raw_text):
        body = m.group("body").strip()
        # body may be "func_name\n{json}", "func_name(args)", "func_name\n<|arg_key|>...",
        # or bare JSON
        sub = _GLM_NAME_NEWLINE_JSON_RE.search(body)
        if sub:
            name = sub.group("name")
            args = _parse_glm_tool_call_args(sub.group("json"), name)
            if args is not None:
                calls.append({"name": name, "arguments": args, "id": "agent"})
                continue
        pm = _GLM_FUNC_CALL_RE.search(body)
        if pm:
            name = pm.group("name")
            args = _parse_glm_tool_call_args(pm.group("args"), name)
            if args is not None:
                calls.append({"name": name, "arguments": args, "id": "agent"})
                continue
        # GLM-4.7-style args: <|arg_key|>k<|/arg_key|><|arg_val|>v<|/arg_val|>
        # or bare value after <|/arg_key|> (no <|arg_val|> wrapper)
        first_line = body.split("\n", 1)[0].strip()
        if first_line in names:
            kv = _GLM_ARG_KV_RE.findall(body)
            if kv:
                args = {}
                for m_kv in kv:
                    k, v = m_kv
                    k = k.strip()
                    if not k:
                        continue
                    args[k] = json.loads(v) if v and v.strip().startswith(("[", "{")) else (v.strip("'\"") if v else "")
                if args:
                    calls.append({"name": first_line, "arguments": args, "id": "agent"})
                continue
            # Fallback: <\|arg_key\>k\nvalue (no closing tag at all)
            bare = _GLM_ARG_BARE_RE.findall(body)
            if bare:
                args = {}
                for k in bare:
                    # Value is everything after the key until the next
                    # <\|arg_key\> or end; split on first newline.
                    parts = k.split("\n", 1)
                    key = parts[0].strip()
                    val = parts[1].strip() if len(parts) > 1 else ""
                    if val.startswith(("<|/arg_key|>", "</arg_key|>")):
                        val = val.split(">", 1)[-1].strip()
                    if key:
                        try:
                            args[key] = json.loads(val) if val and val[0] in "[{" else val
                        except (json.JSONDecodeError, IndexError):
                            args[key] = val
                if args:
                    calls.append({"name": first_line, "arguments": args, "id": "agent"})
                continue
        # bare JSON inside the tag
        if body.startswith("{"):
            try:
                obj = json.loads(body)
                if isinstance(obj, dict) and _json_obj_looks_like_tool_call(obj, names):
                    name = str(obj.get("name") or obj.get("type") or obj.get("function") or "")
                    args = obj.get("arguments") or obj.get("parameters") or {}
                    if name in names:
                        calls.append({"name": name, "arguments": args, "id": "agent"})
            except json.JSONDecodeError:
                pass

    if calls:
        return [calls[-1]]

    # 2. ``func_name(args)`` without a tool_call tag (pythonic, may be in
    # a ```python ... ``` code fence or after a line of reasoning)
    for m in _GLM_FUNC_CALL_RE.finditer(raw_text):
        name = m.group("name")
        args = _parse_glm_tool_call_args(m.group("args"), name)
        if args is not None:
            calls.append({"name": name, "arguments": args, "id": "agent"})
    if calls:
        return [calls[-1]]

    # 3. ``func_name\n{json}`` or ``func_name: {json}`` (with or without a
    # trailing ``` fence — strip code fences first so the closing ``` doesn't
    # block the match)
    for m in _GLM_NAME_NEWLINE_JSON_RE.finditer(raw_text):
        name = m.group("name")
        args = _parse_glm_tool_call_args(m.group("json"), name)
        if args is not None:
            calls.append({"name": name, "arguments": args, "id": "agent"})
    if calls:
        return [calls[-1]]

    # 3b. ``func_name: "value"`` (colon-separated bare string)
    for m in _GLM_NAME_COLON_RE.finditer(raw_text):
        name = m.group("name")
        val = m.group("val")
        if name in ("search_corpus", "grep_corpus"):
            calls.append({"name": name, "arguments": {"query": val}, "id": "agent"})
        else:
            calls.append({"name": name, "arguments": {"value": val}, "id": "agent"})
    if calls:
        return [calls[-1]]

    # 4. bare JSON object containing a tool call — may be at the start, inside a
    # ```json...``` fence, or after a line of reasoning.  Search for the last
    # JSON block (greedy) that looks like a tool call.
    _ARG_TO_TOOL = {
        "query": "search_corpus",
        "queries": "fan_out_search",
        "pattern": "grep_corpus",
        "doc_id": "read_document",
        "doc_ids": "review_docs",
        "add_ids": "curate",
        "remove_ids": "curate",
        "chunk_ids": "review_docs",
    }
    for m in re.finditer(r"\{[^{}]*\{.*?\}\s*\}", raw_text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and _json_obj_looks_like_tool_call(obj, names):
                name = str(obj.get("name") or obj.get("type") or obj.get("function") or "")
                args = obj.get("arguments") or obj.get("parameters") or obj
                if not name and isinstance(args, dict):
                    # No "name" field — infer from argument keys
                    for arg_key in args:
                        if arg_key in _ARG_TO_TOOL:
                            name = _ARG_TO_TOOL[arg_key]
                            break
                if name in names:
                    return [{"name": name, "arguments": args, "id": "agent"}]
        except json.JSONDecodeError:
            pass
    # 4b. single-level JSON object (no nesting)
    stripped = _strip_code_fences(raw_text.strip())
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and _json_obj_looks_like_tool_call(obj, names):
                name = str(obj.get("name") or obj.get("type") or obj.get("function") or "")
                args = obj.get("arguments") or obj.get("parameters") or obj
                if not name and isinstance(args, dict):
                    for arg_key in args:
                        if arg_key in _ARG_TO_TOOL:
                            name = _ARG_TO_TOOL[arg_key]
                            break
                if name in names:
                    return [{"name": name, "arguments": args, "id": "agent"}]
        except json.JSONDecodeError:
            pass

    return []


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
        # The vLLM tool-call parser (e.g. glm45/glm47) failed to extract
        # structured tool_calls, but the response body *does* contain a
        # recognizable tool call.  Try to recover the call(s) from the
        # content text so the eval can proceed instead of always treating
        # this as a hard protocol error (which forces a format retry and
        # yields 0 tool calls / 0 recall for GLM-4-32B-0414).
        recovered = extract_tool_calls_from_content(text)
        if recovered:
            parsed.tool_calls = recovered
            if any(str(c.get("name") or "") == "end_search" for c in parsed.tool_calls):
                parsed.episode_finish_reason = "explicit_end_search"
            return parsed
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
