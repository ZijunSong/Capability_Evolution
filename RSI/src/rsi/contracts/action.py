"""Action protocol: the minimal action space shared by every environment.

An ``Action`` is either a basic ``ToolCall`` (name + JSON arguments, validated
against the environment's frozen basic tool catalog) or a ``Finish``.
Advanced / composite tools are harness-bank components, never actions: the
Student always only sees and produces basic tool calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Union

from rsi.contracts.common import JSONValue


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return {"type": "tool_call", "name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class Finish:
    answer: str | None = None
    artifact_refs: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "type": "finish",
            "answer": self.answer,
            "artifact_refs": list(self.artifact_refs),
        }


Action = Union[ToolCall, Finish]

ACTION_SCHEMA_ID = "rsi_action_v1"

# Canonical textual form used by the H0 action parser.  The model emits one
# JSON object per decision; the parser is strict and versioned.
_ACTION_RE = re.compile(r"^\s*(\{.*\})\s*$", re.DOTALL)


class ActionParseError(ValueError):
    """Model output is not a legal action encoding."""


def encode_action(action: Action) -> str:
    """Canonical text encoding of an action (what the model is asked to emit)."""
    return json.dumps(action.to_json(), sort_keys=True, ensure_ascii=False)


def parse_action(text: str) -> Action:
    """Strict parser for the H0 action protocol.

    Raises :class:`ActionParseError` on any malformed output; the runtime turns
    that into a visible error observation that consumes action budget.
    """
    m = _ACTION_RE.match(text)
    if not m:
        raise ActionParseError("action must be a single JSON object")
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ActionParseError(f"invalid JSON in action: {e}") from e
    return action_from_json(obj)


def action_from_json(obj: Any) -> Action:
    if not isinstance(obj, dict):
        raise ActionParseError("action JSON must be an object")
    typ = obj.get("type")
    if typ == "tool_call":
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            raise ActionParseError("tool_call.name must be a non-empty string")
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            raise ActionParseError("tool_call.arguments must be an object")
        return ToolCall(name=name, arguments=dict(args))
    if typ == "finish":
        answer = obj.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ActionParseError("finish.answer must be a string or null")
        refs = obj.get("artifact_refs", [])
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            raise ActionParseError("finish.artifact_refs must be a list of strings")
        return Finish(answer=answer, artifact_refs=tuple(refs))
    raise ActionParseError(f"unknown action type {typ!r}")


def is_tool_call(action: Action) -> bool:
    return isinstance(action, ToolCall)


def is_finish(action: Action) -> bool:
    return isinstance(action, Finish)
