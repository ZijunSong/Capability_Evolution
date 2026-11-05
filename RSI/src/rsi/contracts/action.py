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
    # AppWorld AnswerType-compatible scalars (None | str | int | float); the
    # adapter submits it verbatim via apis.supervisor.complete_task.
    answer: "str | int | float | None" = None
    artifact_refs: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "type": "finish",
            "answer": self.answer,
            "artifact_refs": list(self.artifact_refs),
        }


Action = Union[ToolCall, Finish]

ACTION_SCHEMA_ID = "rsi_action_v2"

# Canonical textual form used by the H0 action parser.  The model emits one
# JSON object per decision; the parser is strict and versioned.
_ACTION_RE = re.compile(r"^\s*(\{.*\})\s*$", re.DOTALL)


class ActionParseError(ValueError):
    """Model output is not a legal action encoding."""


_THINK_RE = re.compile("\u003cthink\u003e.*?\u003c/think\u003e", re.DOTALL)
_TOOL_RE = re.compile("\u003ctool\u003e.*?\u003c/tool\u003e", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _strip_wrappers(text: str) -> str:
    """Remove thinking/tool-call/markdown wrappers real models emit around the
    action JSON (Qwen thinking blocks, `` tool_calls`` function-call
    style, fenced code)."""
    out = _THINK_RE.sub(" ", text)
    out = _TOOL_RE.sub(" ", out)
    out = out.replace("```json", "```")
    return out


def extract_json_object(text: str) -> dict:
    """Find the single action JSON object inside model output.

    Accepts: bare JSON (canonical), JSON after a thinking block, JSON inside a
    markdown fence.  Raises :class:`ActionParseError` when none is found — the
    protocol still requires exactly one action object.
    """
    cleaned = _strip_wrappers(text)
    m = _FENCE_RE.search(text)
    if m is not None:
        cleaned = _strip_wrappers(text[: m.start()] + " " + m.group(1))
        candidates = [m.group(1)]
    else:
        candidates = []
    # candidate: the full cleaned text
    candidates.append(cleaned.strip())
    # candidate: last balanced {...} region (model may add prose)
    brace_start = cleaned.find("{")
    while brace_start != -1:
        candidates.append(cleaned[brace_start:])
        brace_start = cleaned.find("{", brace_start + 1)
    for cand in candidates:
        c = cand.strip()
        if not c.startswith("{"):
            continue
        # trim trailing junk after the balanced object
        depth = 0
        end = None
        in_str = False
        esc = False
        for i, ch in enumerate(c):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None:
            try:
                obj = json.loads(c[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ActionParseError("no action JSON object found in model output")


def encode_action(action: Action) -> str:
    """Canonical text encoding of an action (what the model is asked to emit)."""
    return json.dumps(action.to_json(), sort_keys=True, ensure_ascii=False)


def parse_action(text: str) -> Action:
    """Strict parser for the H0 action protocol.

    Raises :class:`ActionParseError` on any malformed output; the runtime turns
    that into a visible error observation that consumes action budget.
    """
    obj = extract_json_object(text)
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
        # PR (2026-11-05): AppWorld's AnswerType is None | float | int | str
        # and its GT answer.json stores raw JSON scalars (e.g. 368, null).
        # Rejecting numeric answers forced models to stringify ("368"), which
        # only survives because the official answer_to_text normalizes both
        # sides — accept the numeric scalars natively instead.
        if answer is not None and not isinstance(answer, (str, int, float)):
            raise ActionParseError("finish.answer must be a string, number, or null")
        refs = obj.get("artifact_refs", [])
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            raise ActionParseError("finish.artifact_refs must be a list of strings")
        return Finish(answer=answer, artifact_refs=tuple(refs))
    raise ActionParseError(f"unknown action type {typ!r}")


def is_tool_call(action: Action) -> bool:
    return isinstance(action, ToolCall)


def is_finish(action: Action) -> bool:
    return isinstance(action, Finish)
