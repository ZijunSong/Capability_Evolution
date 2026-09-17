"""Messages / tool-call conversion only (E04).

Does not execute tools, update working memory, retry format errors, or
decide episode end. Those stay on the original ``SlidingWindowSearchEnv``.
"""

from __future__ import annotations

import ast
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

_PYTHONIC_KWARG_NAMES = (
    "add_ids",
    "remove_ids",
    "importance",
    "query",
    "queries",
    "doc_id",
    "doc_ids",
    "pattern",
    "claim",
    "chunk_ids",
    "reasoning",
    "text",
)
_PYTHONIC_KWARG_PATTERN = "|".join(_PYTHONIC_KWARG_NAMES)
_MISSING_PYTHONIC_COMMA_RE = re.compile(
    rf'(?<=[\])}}\d"\'])(?=(?:{_PYTHONIC_KWARG_PATTERN})\s*=)'
)

_TOOL_CALL_IN_CONTENT_RE = re.compile(
    r"(?:<tool_call\b"
    r'|"name"\s*:\s*"(?:' + _TOOL_NAMES_PATTERN + r')"'
    r'|"(?:type|operation|function)"\s*:\s*"(?:' + _TOOL_NAMES_PATTERN + r')"'
    r"|<\|channel\|>|<\|call\|>"
    r"|(?:^|[\s{\[,])(?:" + _TOOL_NAMES_PATTERN + r")\s*[:({]"
    r"|\[(?:\s*(?:" + _TOOL_NAMES_PATTERN + r")\s*\()"
    r"|functions[\.\-\s](?:" + _TOOL_NAMES_PATTERN + r")"
    r"|\b(?:" + _TOOL_NAMES_PATTERN + r")\s*\(\s*\{"
    r"|(?:^|[\s{\[,])(?:" + _TOOL_NAMES_PATTERN + r")\s*\(\s*"
    r"(?:[)\]{]|" + _PYTHONIC_KWARG_PATTERN + r"\s*=)"
    r"|(?:^|\n)(?:" + _TOOL_NAMES_PATTERN + r")\s*\n\s*\{"
    r")",
    re.IGNORECASE,
)

_GLM0414_NAME_JSON_RE = re.compile(
    r"^(" + _TOOL_NAMES_PATTERN + r")\s*\n\s*(\{.*\})\s*$",
    re.IGNORECASE | re.DOTALL,
)
_GLM0414_NAME_HEAD_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s`(\[{]))(" + _TOOL_NAMES_PATTERN + r")(?=\s*(?:\n\s*\{|\())",
    re.IGNORECASE,
)
_FENCE_BLOCK_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[^\n`]*\n?")
_COLON_KWARG_RE = re.compile(r"\b(" + _PYTHONIC_KWARG_PATTERN + r")\s*:")
_POSITIONAL_TOOLS = frozenset({"search_corpus", "grep_corpus", "fan_out_search"})

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


def _unwrap_markdown_fences(text: str) -> str:
    """Drop markdown fences so buried GLM/pythonic calls are visible to scanners."""
    raw = str(text or "")
    prev = None
    while prev != raw:
        prev = raw
        raw = _FENCE_BLOCK_RE.sub(lambda m: "\n" + m.group(1).strip() + "\n", raw)
    return _OPEN_FENCE_RE.sub("\n", raw).strip()


def _extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> str | None:
    if start >= len(text) or text[start] != open_ch:
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
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _dedupe_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("name") or "")
        try:
            args_key = json.dumps(call.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
        except TypeError:
            args_key = repr(call.get("arguments"))
        key = (name, args_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def _json_args(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _positional_arguments(name: str, inner: str) -> dict[str, Any] | None:
    text = inner.strip()
    if not text:
        return {} if name == "end_search" else None
    if name == "search_corpus":
        return {"query": text}
    if name == "grep_corpus":
        return {"pattern": text}
    if name == "fan_out_search":
        return {"queries": [text]}
    return None


def _call_from_paren_body(name: str, inner: str, names: frozenset[str]) -> dict[str, Any] | None:
    body = inner.strip()
    if body.startswith("{") and body.endswith("}"):
        args = _json_args(body)
        if args is not None:
            return {"name": name, "arguments": args, "id": "agent"}
    normalized = _COLON_KWARG_RE.sub(r"\1=", body)
    snippet = f"{name}({normalized})"
    try:
        tree = ast.parse(_insert_missing_pythonic_commas(snippet), mode="eval")
    except SyntaxError:
        tree = None
    if tree is not None and isinstance(tree.body, ast.Call):
        item = _pythonic_call_to_tool(tree.body, names)
        if item is not None:
            return item
        if (
            name in _POSITIONAL_TOOLS
            and isinstance(tree.body.func, ast.Name)
            and len(tree.body.args) == 1
            and not tree.body.keywords
        ):
            arg = tree.body.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.strip():
                mapped = _positional_arguments(name, arg.value)
                if mapped is not None:
                    return {"name": name, "arguments": mapped, "id": "agent"}
            if isinstance(arg, ast.List) and name == "fan_out_search":
                try:
                    queries = ast.literal_eval(arg)
                except (ValueError, TypeError, SyntaxError, MemoryError):
                    queries = None
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    return {"name": name, "arguments": {"queries": queries}, "id": "agent"}
    if name not in _POSITIONAL_TOOLS:
        return None
    if _COLON_KWARG_RE.search(body) or "=" in body:
        return None
    if len(body.split()) < 2 and not (body.startswith('"') or body.startswith("'")):
        return None
    if body.startswith(('"', "'")) and body.endswith(('"', "'")) and len(body) >= 2:
        body = body[1:-1]
    mapped = _positional_arguments(name, body)
    if mapped is None:
        return None
    return {"name": name, "arguments": mapped, "id": "agent"}


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
    if _try_parse_pythonic_tool_calls(raw, names) is not None:
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


def _insert_missing_pythonic_commas(src: str) -> str:
    """Gemma often emits [curate(add_ids=[...]importance={...})] without commas."""
    return _MISSING_PYTHONIC_COMMA_RE.sub(",", src)


def _pythonic_call_to_tool(call: ast.Call, names: frozenset[str]) -> dict[str, Any] | None:
    if not isinstance(call.func, ast.Name):
        return None
    name = call.func.id
    if name not in names or name == "user_text":
        return None
    if call.args or any(kw.arg is None for kw in call.keywords):
        return None
    arguments: dict[str, Any] = {}
    for kw in call.keywords:
        assert kw.arg is not None
        try:
            arguments[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError, SyntaxError, MemoryError):
            return None
    return {"name": name, "arguments": arguments, "id": "agent"}


def _try_parse_glm0414_tool_calls(
    text: str, known_tool_names: frozenset[str] | None = None
) -> list[dict[str, Any]] | None:
    """Recover GLM-4-0414 calls leaked into mixed assistant content.

    Handles native ``name\\n{json}``, pythonic ``name(kw=...)``, colon kwargs
    ``name(query: "...")``, and a quoted/multi-word positional query. Planning
    prose around the call is ignored. Only known tool names are accepted so a
    markdown ``json`` fence is never promoted into ``invalid_tool_name``.
    """
    names = known_tool_names or _KNOWN_TOOL_NAMES
    raw = _unwrap_markdown_fences(str(text or "").strip())
    if not raw:
        return None
    match = _GLM0414_NAME_JSON_RE.match(raw)
    if match:
        name = match.group(1)
        args = _json_args(match.group(2)) if name in names else None
        if args is not None:
            return [{"name": name, "arguments": args, "id": "agent"}]
    recovered: list[dict[str, Any]] = []
    for match in _GLM0414_NAME_HEAD_RE.finditer(raw):
        name = match.group(1)
        if name not in names:
            continue
        rest = raw[match.end() :].lstrip()
        if rest.startswith("{"):
            blob = _extract_balanced(rest, 0, "{", "}")
            args = _json_args(blob) if blob else None
            if args is not None:
                recovered.append({"name": name, "arguments": args, "id": "agent"})
            continue
        if rest.startswith("("):
            blob = _extract_balanced(rest, 0, "(", ")")
            if not blob:
                continue
            item = _call_from_paren_body(name, blob[1:-1], names)
            if item is not None:
                recovered.append(item)
    recovered = _dedupe_tool_calls(recovered)
    return recovered or None


def _try_parse_pythonic_tool_calls(
    text: str, known_tool_names: frozenset[str] | None = None
) -> list[dict[str, Any]] | None:
    """Recover Gemma pythonic tool calls written into message content.

    vLLM's pythonic parser drops malformed list-calls (missing commas between
    kwargs) into ``content``. At temperature 0 a format retry usually repeats
    the same string, so recover the intended tool call instead of ending the
    episode as ``implicit_user_text``.
    """
    names = known_tool_names or _KNOWN_TOOL_NAMES
    raw = _insert_missing_pythonic_commas(_strip_code_fences(str(text or "").strip()))
    if not raw:
        return None
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError:
        return None
    body = tree.body
    if isinstance(body, ast.List):
        elts = body.elts
    elif isinstance(body, ast.Tuple):
        elts = body.elts
    elif isinstance(body, ast.Call):
        elts = [body]
    else:
        return None
    if not elts:
        return None
    recovered: list[dict[str, Any]] = []
    for elt in elts:
        if not isinstance(elt, ast.Call):
            return None
        item = _pythonic_call_to_tool(elt, names)
        if item is None:
            return None
        recovered.append(item)
    return recovered


def _apply_recovered_tool_calls(parsed: ParsedApiAction, calls: list[dict[str, Any]]) -> ParsedApiAction:
    parsed.tool_calls = calls
    parsed.parse_error = None
    parsed.protocol_error = None
    parsed.episode_finish_reason = (
        "explicit_end_search"
        if any(str(c.get("name") or "") == "end_search" for c in calls)
        else None
    )
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
    recovered = _try_parse_pythonic_tool_calls(text) if text else None
    if recovered:
        return _apply_recovered_tool_calls(parsed, recovered)
    recovered = _try_parse_glm0414_tool_calls(text) if text else None
    if recovered:
        return _apply_recovered_tool_calls(parsed, recovered)
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


def _is_glm0414_model_name(model: str | None) -> bool:
    if not model:
        return False
    from trim.eval.model_profiles import GLM0414_PROFILE, classify_profile_by_name

    return classify_profile_by_name(str(model)) is GLM0414_PROFILE


def rewrite_premature_user_text(
    parsed: ParsedApiAction, *, env_turn: int, model: str | None = None
) -> ParsedApiAction:
    """Treat first-turn prose as a format error for retrieval subagents.

    BrowseComp zero/all actors must call search/curate/end_search. Mapping the
    first completion to ``user_text`` ends the episode with zero retrieval.
    Later-turn pythonic tool text that leaked through as ``user_text`` is also
    recovered or retried — otherwise Gemma all dies after the second search.
    GLM-4-0414 additionally retries later-turn natural language: after a
    successful search it narrates "I should curate now" and would otherwise
    finish as ``implicit_user_text`` with mean_turns≈1.
    """
    if parsed.episode_finish_reason != "implicit_user_text":
        return parsed
    text = ""
    if parsed.tool_calls:
        args = parsed.tool_calls[0].get("arguments") or {}
        if isinstance(args, Mapping):
            text = str(args.get("text") or "")
    recovered = _try_parse_pythonic_tool_calls(text) if text else None
    if recovered:
        return _apply_recovered_tool_calls(parsed, recovered)
    recovered = _try_parse_glm0414_tool_calls(text) if text else None
    if recovered:
        return _apply_recovered_tool_calls(parsed, recovered)
    if text and _content_looks_like_tool_call(text):
        parsed.parse_error = "Response body contains tool-call syntax without structured tool_calls"
        parsed.protocol_error = "tool_call_in_content"
        parsed.episode_finish_reason = None
        parsed.tool_calls = []
        return parsed
    if int(env_turn) > 0 and not _is_glm0414_model_name(model):
        return parsed
    parsed.parse_error = "Retrieval subagent returned natural language instead of a tool call"
    parsed.protocol_error = parsed.protocol_error or "premature_user_text"
    parsed.episode_finish_reason = None
    parsed.tool_calls = []
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
