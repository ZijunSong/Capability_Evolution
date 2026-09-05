"""Encode Harmony conversations with the gpt-oss Hugging Face tokenizer.

openai_harmony's Rust encoding downloads o200k_base.tiktoken from Azure.
Offline / approved runtimes often fail that download, and the previous
TRIM fallback rendered ``[Role.SYSTEM] {...}`` as character ordinals. Those
ASCII IDs were then sent to a real gpt-oss vLLM worker, so every tool call
parsed as ``unknown``.

The checkpoint tokenizer already has the Harmony specials
(``<|start|>=200006``, ``<|call|>=200012``). Rendering the conversation to
Harmony text and encoding with that tokenizer produces the same IDs as
``HarmonyGptOss`` when the vocab is reachable.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

HARMONY_START_ID = 200006  # <|start|>
HARMONY_END_ID = 200007  # <|end|>
HARMONY_MESSAGE_ID = 200008  # <|message|>
HARMONY_CALL_ID = 200012  # <|call|>
HARMONY_RETURN_ID = 200002  # <|return|>


def _role_value(role: Any) -> str:
    if role is None:
        return ""
    value = getattr(role, "value", None)
    if value:
        return str(value).lower()
    text = str(role)
    if text.startswith("Role."):
        text = text.split(".", 1)[1]
    return text.lower()


def _messages_from_conversation(conversation: Any) -> list[dict[str, Any]]:
    if conversation is None:
        return []
    if hasattr(conversation, "to_json"):
        blob = json.loads(conversation.to_json())
        return list(blob.get("messages") or [])
    if hasattr(conversation, "to_dict"):
        blob = conversation.to_dict()
        return list(blob.get("messages") or [])
    messages = getattr(conversation, "messages", None)
    if messages is None and isinstance(conversation, (list, tuple)):
        messages = conversation
    out: list[dict[str, Any]] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            out.append(msg)
        elif hasattr(msg, "to_dict"):
            out.append(msg.to_dict())
        else:
            out.append(
                {
                    "role": _role_value(getattr(msg, "role", None)),
                    "content": [{"type": "text", "text": str(getattr(msg, "content", "") or "")}],
                }
            )
    return out


def _content_dicts(msg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = msg.get("content")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"type": "text", "text": raw}]
    if isinstance(raw, dict):
        return [raw]
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        elif hasattr(item, "to_dict"):
            out.append(item.to_dict())
        elif hasattr(item, "text"):
            out.append({"type": "text", "text": str(item.text)})
        else:
            out.append({"type": "text", "text": str(item)})
    return out


def _text_of(msg: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in _content_dicts(msg):
        if item.get("type") in {None, "text"} and "text" in item:
            parts.append(str(item.get("text") or ""))
        elif item.get("type") == "system_content":
            parts.append(_render_system_content(item, has_function_tools=True))
        elif item.get("type") == "developer_content":
            parts.append(_render_developer_content(item))
    return "".join(parts)


def _render_system_content(content: dict[str, Any], *, has_function_tools: bool) -> str:
    identity = content.get("model_identity") or (
        "You are ChatGPT, a large language model trained by OpenAI."
    )
    cutoff = content.get("knowledge_cutoff") or "2024-06"
    date = content.get("conversation_start_date") or "2026-04-01"
    effort = content.get("reasoning_effort") or "high"
    if hasattr(effort, "value"):
        effort = effort.value
    effort = str(effort).lower()
    cfg = content.get("channel_config") or {}
    channels = cfg.get("valid_channels") or ["analysis", "commentary", "final"]
    lines = [
        str(identity),
        f"Knowledge cutoff: {cutoff}",
        f"Current date: {date}",
        "",
        f"Reasoning: {effort}",
        "",
        "# Valid channels: "
        + ", ".join(str(c) for c in channels)
        + ". Channel must be included for every message.",
    ]
    text = "\n".join(lines)
    if has_function_tools or content.get("tools"):
        text += "\nCalls to these tools must go to the commentary channel: 'functions'."
    return text


def _ts_type(spec: dict[str, Any] | None, depth: int = 0) -> str:
    spec = spec or {}
    kind = spec.get("type")
    if kind == "array":
        items = spec.get("items") or {}
        return _ts_type(items, depth) + "[]"
    if kind == "string":
        return "string"
    if kind in {"number", "integer"}:
        return "number"
    if kind == "boolean":
        return "boolean"
    if kind == "object" or spec.get("properties") is not None:
        props = spec.get("properties")
        pad = " " * (depth * 4)
        if not props:
            return "{\n" + pad + "}"
        required = set(spec.get("required") or [])
        lines = ["{"]
        for name, child in props.items():
            child = child or {}
            desc = child.get("description")
            if desc:
                lines.append(f"{pad}// {desc}")
            optional = "" if name in required else "?"
            lines.append(
                f"{pad}{name}{optional}: {_ts_type(child, depth + 1)},"
            )
        lines.append(pad + "}")
        return "\n".join(lines)
    return "any"


def _render_tool_namespace(name: str, tools: list[dict[str, Any]]) -> str:
    chunks = [f"## {name}", "", f"namespace {name} {{", ""]
    for tool in tools:
        desc = str(tool.get("description") or "").strip()
        if desc:
            chunks.append(f"// {desc}")
        params = tool.get("parameters") or {}
        chunks.append(f"type {tool.get('name')} = (_: {_ts_type(params, 0)}) => any;")
        chunks.append("")
    chunks.append(f"}} // namespace {name}")
    return "\n".join(chunks)


def _render_developer_content(content: dict[str, Any]) -> str:
    parts: list[str] = []
    instructions = content.get("instructions")
    if instructions:
        parts.append("# Instructions")
        parts.append("")
        parts.append(str(instructions).rstrip())
        parts.append("")
    tools_map = content.get("tools") or {}
    if tools_map:
        parts.append("# Tools")
        parts.append("")
        for ns_name, ns in tools_map.items():
            ns_tools = ns.get("tools") if isinstance(ns, dict) else None
            if ns_tools is None and hasattr(ns, "tools"):
                ns_tools = ns.tools
            rendered = _render_tool_namespace(str(ns_name), list(ns_tools or []))
            parts.append(rendered)
    return "\n".join(parts).rstrip()


def _conversation_has_function_tools(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        for item in _content_dicts(msg):
            tools = item.get("tools")
            if isinstance(tools, dict) and tools.get("functions"):
                return True
    return False


def _message_body(msg: dict[str, Any], *, has_function_tools: bool) -> str:
    parts: list[str] = []
    for item in _content_dicts(msg):
        kind = item.get("type")
        if kind == "system_content":
            parts.append(_render_system_content(item, has_function_tools=has_function_tools))
        elif kind == "developer_content":
            parts.append(_render_developer_content(item))
        elif "text" in item:
            parts.append(str(item.get("text") or ""))
    return "".join(parts)


def _is_tool_call(msg: dict[str, Any]) -> bool:
    recipient = str(msg.get("recipient") or "")
    content_type = str(msg.get("content_type") or "")
    return recipient.startswith("functions.") or "constrain" in content_type


def _start_header(msg: dict[str, Any]) -> str:
    role = str(msg.get("role") or "assistant")
    name = msg.get("name")
    recipient = msg.get("recipient")
    channel = msg.get("channel")
    content_type = msg.get("content_type")
    if role == "tool" and name:
        head = f"<|start|>{name}"
    else:
        head = f"<|start|>{role}"
    if recipient:
        head += f" to={recipient}"
    if channel:
        head += f"<|channel|>{channel}"
    if content_type:
        ctype = str(content_type)
        if channel:
            head += f" {ctype}" if not ctype.startswith(" ") else ctype
        else:
            head += ctype
    return head + "<|message|>"


def render_harmony_conversation_text(
    conversation: Any,
    *,
    next_role: Any = None,
) -> str:
    """Render a Harmony conversation to the gpt-oss control-token string."""
    messages = _messages_from_conversation(conversation)
    has_fn = _conversation_has_function_tools(messages)
    chunks: list[str] = []
    for msg in messages:
        body = _message_body(msg, has_function_tools=has_fn)
        end = "<|call|>" if _is_tool_call(msg) else "<|end|>"
        chunks.append(_start_header(msg) + body + end)
    if next_role:
        chunks.append(f"<|start|>{_role_value(next_role)}")
    return "".join(chunks)


class TokenizerHarmonyEncoding:
    """Duck-typed Harmony encoding backed by a gpt-oss AutoTokenizer."""

    name = "HarmonyGptOss"

    def __init__(self, tokenizer: Any, *, source: str = "") -> None:
        self.tokenizer = tokenizer
        self.source = source

    @classmethod
    def from_pretrained(cls, model_path: str) -> "TokenizerHarmonyEncoding":
        from trim.eval.model_tokenizer import patch_transformers_tokenizer_compat
        from trim.training.vllm_hybrid import assert_gptoss_tokenizer
        from transformers import AutoTokenizer

        patch_transformers_tokenizer_compat()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        assert_gptoss_tokenizer(tokenizer, source=model_path)
        return cls(tokenizer, source=model_path)

    def stop_tokens_for_assistant_actions(self) -> list[int]:
        return [HARMONY_CALL_ID, HARMONY_RETURN_ID]

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return [int(x) for x in self.tokenizer.encode(text, add_special_tokens=False)]

    def decode_utf8(self, tokens: Iterable[int]) -> str:
        return str(
            self.tokenizer.decode(list(tokens), skip_special_tokens=False)
        )

    def decode(self, tokens: Iterable[int]) -> str:
        return self.decode_utf8(tokens)

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        return self.decode_utf8(tokens)

    def render_conversation_for_completion(
        self,
        conversation: Any,
        next_turn_role: Any = None,
        config: Any = None,
    ) -> list[int]:
        del config
        text = render_harmony_conversation_text(
            conversation, next_role=next_turn_role
        )
        return self.encode(text)

    def render_conversation(self, conversation: Any, config: Any = None) -> list[int]:
        del config
        return self.render_conversation_for_completion(conversation, next_turn_role=None)

    def parse_messages_from_completion_tokens(
        self,
        tokens: Sequence[int],
        role: Any = None,
        *,
        strict: bool = True,
    ):
        del role, strict
        raise RuntimeError(
            "TokenizerHarmonyEncoding does not wrap rust parse_messages; "
            "TRIM falls back to the Harmony text parser."
        )
