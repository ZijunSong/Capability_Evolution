"""Harness-G prompt + action parse (named tools, not ephemeral A0 ids).

Policy interface inside TRIM is Harmony-like ``to=select {json}``. The
upstream Harness-G A0 menu is mapped onto select / lookup / answer /
answer_with so OPD can score a stable action name.

gpt-oss / vLLM requires an o200k Harmony conversation (first token
``<|start|>=200006``). Qwen3 still uses the HF chat template. Never encode
the plain-text ``render_prompt()`` string with a gpt-oss encoder.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from trim.training.action_codec import parse_action as parse_codec_action

_TO_RE = re.compile(
    r"to=(?:functions\.)?(?P<name>select|lookup|answer|answer_with|init)\b",
    re.I,
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_AID_RE = re.compile(r"\b(A\d+)\b", re.I)

SYSTEM_PROMPT = """You are a Harness-G search agent.
Basic runtime tools (always available):
- init: first retrieve visible evidence sentences
- select: commit a visible sentence sid as evidence
- lookup: follow an entity eid; the environment builds the retrieval query
- answer: stop and answer from selected evidence
Do not invent free-form search queries. Pick a tool and its target id.
"""

HARNESS_G_FUNCTION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "init",
        "description": "First retrieve visible evidence sentences from the document store.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "select",
        "description": "Commit a visible sentence sid as evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "sid": {"type": "string", "description": "Visible sentence id to commit."}
            },
            "required": ["sid"],
        },
    },
    {
        "name": "lookup",
        "description": "Follow an entity eid; the environment builds the retrieval query.",
        "parameters": {
            "type": "object",
            "properties": {
                "eid": {"type": "string", "description": "Entity id to follow."}
            },
            "required": ["eid"],
        },
    },
    {
        "name": "answer",
        "description": "Stop and answer from selected evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Short reason for stopping."}
            },
            "required": [],
        },
    },
)
ANSWER_WITH_TOOL: dict[str, Any] = {
    "name": "answer_with",
    "description": "Stop and answer using one selected sentence sid as the cited evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "sid": {"type": "string", "description": "Selected sentence id to cite."}
        },
        "required": ["sid"],
    },
}


def render_prompt(query: str, wm_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + f"\nQuestion: {query}\n"
        + (wm_text or "")
        + "\nEmit one tool call: to=select|lookup|answer {json}\n"
    )


def _is_qwen_family(enc: Any) -> bool:
    family = str(getattr(enc, "family", "") or "").lower()
    return family in {"qwen3", "qwen3_chat", "qwen"}


def _harmony_backend(enc: Any) -> Any | None:
    if enc is None:
        return None
    inner = getattr(enc, "harmony", None)
    if inner is not None and hasattr(inner, "render_conversation_for_completion"):
        return inner
    if hasattr(enc, "render_conversation_for_completion"):
        return enc
    return None


def _include_answer_with(harness_mask: Mapping[str, Any] | None) -> bool:
    if not harness_mask:
        return False
    return bool(harness_mask.get("answer_with"))


def _tool_specs(*, include_answer_with: bool) -> list[dict[str, Any]]:
    specs = [dict(item) for item in HARNESS_G_FUNCTION_TOOLS]
    if include_answer_with:
        specs.append(dict(ANSWER_WITH_TOOL))
    return specs


def build_harness_g_conversation(
    query: str,
    wm_text: str,
    *,
    include_answer_with: bool = False,
) -> Any:
    """Harmony Conversation with Harness-G function tools (not Harness-1 tools)."""
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
        ToolDescription,
    )

    tools = [
        ToolDescription.new(str(spec["name"]), str(spec["description"]), spec["parameters"])
        for spec in _tool_specs(include_answer_with=include_answer_with)
    ]
    system = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2026-04-01")
    )
    developer = DeveloperContent.new().with_function_tools(tools)
    user_text = render_prompt(query, wm_text)
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.DEVELOPER, developer),
        Message.from_role_and_content(Role.USER, user_text),
    ]
    return Conversation(messages=messages)


def parse_harness_g_action(
    text: str,
    *,
    action_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool]:
    blob = str(text or "").strip()
    if not blob:
        return {"name": "unknown", "arguments": {}}, False
    try:
        parsed = parse_codec_action(blob)
        name = str(parsed.get("name") or "").lower()
        if name in {"select", "lookup", "answer", "answer_with", "init"}:
            return {"name": name, "arguments": dict(parsed.get("arguments") or {})}, True
    except Exception:
        pass
    match = _TO_RE.search(blob)
    if match:
        name = match.group("name").lower()
        args: dict[str, Any] = {}
        jm = _JSON_RE.search(blob[match.end() :])
        if jm:
            try:
                loaded = json.loads(jm.group(0))
                if isinstance(loaded, dict):
                    args = loaded
            except json.JSONDecodeError:
                args = {}
        return {"name": name, "arguments": args}, True
    aid = _AID_RE.search(blob)
    if aid and action_map:
        mapped = action_map.get(aid.group(1)) or action_map.get(aid.group(1).upper())
        if mapped:
            name = str(mapped.get("name") or mapped.get("type") or "").lower()
            args = {}
            if mapped.get("sid"):
                args["sid"] = mapped["sid"]
            if mapped.get("eid"):
                args["eid"] = mapped["eid"]
            if mapped.get("sids"):
                args["sids"] = list(mapped["sids"])
            if name:
                return {"name": name, "arguments": args}, True
    return {"name": "unknown", "arguments": {}}, False


def build_prompt_ids(
    query: str,
    wm_text: str,
    enc,
    *,
    harness_mask: Mapping[str, Any] | None = None,
) -> list[int]:
    prompt = render_prompt(query, wm_text)
    if enc is None:
        return []
    tokenizer = getattr(enc, "tokenizer", None)
    if _is_qwen_family(enc) and tokenizer is not None:
        try:
            raw = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        except TypeError:
            raw = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        from trim.eval.model_tokenizer import _to_token_ids, assert_qwen3_prompt_ids

        return assert_qwen3_prompt_ids(_to_token_ids(raw), what="Harness-G Qwen3 prompt")
    harmony = _harmony_backend(enc)
    if harmony is None:
        raise RuntimeError(
            "Harness-G gpt-oss prompt requires Harmony "
            "render_conversation_for_completion; raw-text encode() is forbidden "
            "because it does not emit <|start|>=200006."
        )
    from openai_harmony import Role
    from trim.eval.harmony_runtime import assert_o200k_harmony_token_ids

    conv = build_harness_g_conversation(
        query,
        wm_text,
        include_answer_with=_include_answer_with(harness_mask),
    )
    try:
        ids = harmony.render_conversation_for_completion(conv, Role.ASSISTANT)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Harness-G Harmony renderer failed: {type(exc).__name__}: {exc}"
        ) from exc
    return assert_o200k_harmony_token_ids(ids, what="Harness-G Harmony prompt")
