from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any, Iterable, List


class _TextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _ParsedMessage:
    def __init__(self, channel: str, text: str, recipient: str | None = None) -> None:
        self.channel = channel
        self.recipient = recipient
        self.content = [_TextContent(text)]


class _LocalRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"


class _LocalReasoningEffort:
    HIGH = "high"


class _LocalAuthor:
    def __init__(self, role: str, name: str | None = None) -> None:
        self.role = role
        self.name = name


class _LocalContent:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {}

    @classmethod
    def new(cls):
        return cls()

    def with_reasoning_effort(self, value: Any):
        self.payload["reasoning_effort"] = value
        return self

    def with_conversation_start_date(self, value: Any):
        self.payload["conversation_start_date"] = value
        return self

    def with_function_tools(self, value: Any):
        self.payload["function_tools"] = value
        return self

    def model_dump(self):
        return self.payload


class _LocalToolDescription:
    def __init__(self, name: str, description: str, parameters: Any) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters

    @classmethod
    def new(cls, name: str, description: str, parameters: Any):
        return cls(name, description, parameters)

    def model_dump(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class _LocalMessage:
    def __init__(self, role: str | None = None, content: Any = None, author: Any = None) -> None:
        self.role = role
        self.author = author
        self.content = [content if hasattr(content, "text") else _TextContent(str(content))]
        self.channel = None
        self.recipient = None

    @classmethod
    def from_role_and_content(cls, role: str, content: Any):
        return cls(role=role, content=content)

    @classmethod
    def from_author_and_content(cls, author: Any, content: Any):
        return cls(role=getattr(author, "role", None), author=author, content=content)

    def with_channel(self, channel: str):
        self.channel = channel
        return self

    def with_recipient(self, recipient: str):
        self.recipient = recipient
        return self

    def with_content_type(self, content_type: str):
        self.content_type = content_type
        return self


class _LocalConversation:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


class _LocalHarmonyEncodingFallback:
    def __init__(self) -> None:
        self._enc = None
        if os.environ.get("SCAPE_DISABLE_TIKTOKEN_FALLBACK", "0") == "1":
            return
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(os.environ.get("SCAPE_TIKTOKEN_FALLBACK", "cl100k_base"))
        except Exception:
            self._enc = None

    def _content_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if hasattr(content, "text"):
            return str(content.text)
        if isinstance(content, (list, tuple)):
            return "\n".join(self._content_text(item) for item in content)
        if hasattr(content, "model_dump"):
            try:
                dumped = content.model_dump()
                if isinstance(dumped, dict) and "function_tools" in dumped:
                    tools = []
                    for tool in dumped.get("function_tools") or []:
                        if hasattr(tool, "model_dump"):
                            tool = tool.model_dump()
                        if isinstance(tool, dict):
                            tools.append({
                                "name": tool.get("name"),
                                "description": tool.get("description"),
                                "parameters": tool.get("parameters"),
                            })
                    dumped = {**dumped, "function_tools": tools}
                return json.dumps(dumped, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                pass
        return repr(content)

    def render_conversation(self, conversation: Any, config: Any = None) -> List[int]:
        parts: list[str] = []
        messages = getattr(conversation, "messages", None)
        if messages is None and isinstance(conversation, (list, tuple)):
            messages = conversation
        if messages is not None:
            for msg in messages:
                role = getattr(msg, "role", None) or getattr(getattr(msg, "author", None), "role", None) or "message"
                channel = getattr(msg, "channel", None)
                recipient = getattr(msg, "recipient", None)
                label = str(role)
                if channel:
                    label += f"/{channel}"
                if recipient:
                    label += f"->{recipient}"
                parts.append(f"[{label}] {self._content_text(getattr(msg, 'content', None))}")
            text = "\n".join(parts)
        else:
            text = repr(conversation)
        if self._enc is not None:
            try:
                return self._enc.encode(text)
            except Exception:
                pass
        return [ord(ch) % 65535 for ch in text]

    def decode_utf8(self, tokens: Iterable[int]) -> str:
        token_list = [int(t) for t in tokens]
        if self._enc is not None:
            try:
                return self._enc.decode(token_list)
            except Exception:
                pass
        return "".join(chr(t % 65535) for t in token_list)

    def stop_tokens_for_assistant_actions(self) -> List[int]:
        return [200002, 200012]

    def parse_messages_from_completion_tokens(self, tokens: List[int]):
        text = self.decode_utf8(tokens).strip()
        if not text:
            return []
        parsed = self._extract_tool_payload(text)
        if parsed is not None:
            recipient, payload = parsed
            return [_ParsedMessage("commentary", json.dumps(payload, ensure_ascii=False), recipient=recipient)]
        return [_ParsedMessage("final", text)]

    def _extract_tool_payload(self, text: str):
        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced)
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            candidates.append(text[brace_start:brace_end + 1])
        for candidate in candidates:
            try:
                obj = json.loads(candidate.strip())
            except Exception:
                continue
            if isinstance(obj, dict) and "tool_calls" in obj:
                return "functions.multi_tool_use", obj
            if isinstance(obj, list):
                return "functions.multi_tool_use", {"tool_calls": obj}
            if isinstance(obj, dict):
                name = obj.get("tool_name") or obj.get("name") or obj.get("tool")
                params = obj.get("parameters") or obj.get("arguments") or obj.get("args")
                if name and isinstance(params, dict):
                    return f"functions.{name}", params
        match = re.search(r"functions\.([a-zA-Z0-9_]+)\s*(\{.*\})", text, flags=re.DOTALL)
        if match:
            try:
                return f"functions.{match.group(1)}", json.loads(match.group(2))
            except Exception:
                return None
        return None
