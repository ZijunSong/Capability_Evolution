"""Harness-G prompt + action parse (named tools, not ephemeral A0 ids).

Policy interface inside TRIM is Harmony-like ``to=select {json}``. The
upstream Harness-G A0 menu is mapped onto select / lookup / answer /
answer_with so OPD can score a stable action name.

gpt-oss / vLLM requires an o200k Harmony conversation (first token
``<|start|>=200006``). Qwen3 still uses the HF chat template with Harness-G
function tools registered. Never encode the plain-text ``render_prompt()``
string with a gpt-oss encoder.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from trim.training.action_codec import parse_action as parse_codec_action

_HARNESS_G_TOOL_NAMES = frozenset({"init", "select", "lookup", "answer", "answer_with"})
_TO_RE = re.compile(
    r"to=(?:functions\.)?(?P<name>select|lookup|answer|answer_with|init)\b",
    re.I,
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_AID_RE = re.compile(r"\b(A\d+)\b", re.I)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")

SYSTEM_PROMPT = """You are a Harness-G search agent.
Basic runtime tools (always available):
- init: first retrieve visible evidence sentences
- select: commit a visible sentence sid as evidence (use exact sid="..." from working memory)
- lookup: follow an entity eid (use exact eid="..." from working memory; environment builds retrieval query)
- answer: stop and answer from selected evidence
Rules:
- Use ONLY target ids shown in working memory (sid=..., eid=...). Never substitute surface names for eid.
- Do not invent free-form search queries or new ids.
- Every turn emit exactly ONE executable tool call with valid JSON arguments.
- Do not describe tools in analysis prose; emit the actual tool call region only.
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
        + "\nEmit one tool call using init, select, lookup, or answer.\n"
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


def qwen_harness_g_tool_schemas(*, include_answer_with: bool = False) -> list[dict[str, Any]]:
    """OpenAI-style tool schemas for Qwen3 chat templates."""
    out: list[dict[str, Any]] = []
    for spec in _tool_specs(include_answer_with=include_answer_with):
        out.append(
            {
                "type": "function",
                "function": {
                    "name": str(spec["name"]),
                    "description": str(spec["description"]),
                    "parameters": dict(spec["parameters"]),
                },
            }
        )
    return out


def _action_name_args(action: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(action, dict):
        return str(action.get("name") or ""), dict(action.get("arguments") or {})
    tools = getattr(action, "tools", None) or []
    params = getattr(action, "params", None) or []
    name = ""
    if tools:
        schema = getattr(tools[0], "tool_schema", None)
        name = str(getattr(schema, "name", "") or "")
    args = params[0] if params else {}
    return name, dict(args or {})


def _obs_text(obs: Any) -> str:
    if isinstance(obs, str):
        return obs
    parts = getattr(obs, "observations", None)
    if parts:
        return "\n".join(str(x) for x in parts)
    return str(obs)


def _qwen_messages(
    query: str,
    wm_text: str,
    actions_obs: list[tuple[Any, Any]] | None,
    *,
    include_answer_with: bool,
) -> list[dict[str, Any]]:
    from trim.eval.harmony_runtime import recent_actions_obs

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}"},
    ]
    if wm_text:
        messages.append({"role": "user", "content": str(wm_text)})
    for action, obs in recent_actions_obs(list(actions_obs or []), keep=12):
        name, args = _action_name_args(action)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": _obs_text(obs)})
    if actions_obs:
        messages.append(
            {
                "role": "user",
                "content": "Continue the Harness-G search. Emit exactly one tool call.",
            }
        )
    return messages


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


def build_harness_g_context(
    query: str,
    wm_text: str,
    actions_obs: list[tuple[Any, Any]] | None = None,
    *,
    include_answer_with: bool = False,
) -> Any:
    """Harmony context with Harness-G tools and recent tool-call history."""
    from openai_harmony import Conversation, DeveloperContent, Message, ReasoningEffort, Role, SystemContent, ToolDescription
    from trim.eval.harmony_runtime import _ensure_scope, make_action, make_observation, recent_actions_obs

    _ensure_scope()
    from harness.ultra_core import action_observation_to_messages

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
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.DEVELOPER, developer),
        Message.from_role_and_content(Role.USER, f"Question: {query}"),
    ]
    if wm_text:
        messages.append(Message.from_role_and_content(Role.USER, str(wm_text)))
    for action, obs in recent_actions_obs(list(actions_obs or []), keep=12):
        if hasattr(action, "tools"):
            act_obj = action
            obs_obj = obs
        else:
            name, args = _action_name_args(action)
            act_obj = make_action(name, args)
            obs_obj = obs if hasattr(obs, "observations") else make_observation(_obs_text(obs))
        messages.extend(action_observation_to_messages(act_obj, obs_obj, compress=False))
    if actions_obs:
        messages.append(
            Message.from_role_and_content(
                Role.USER,
                "Continue the Harness-G search. Emit exactly one tool call using the registered functions.",
            )
        )
    return Conversation(messages=messages)


def _parse_json_harness_g_action(blob: str) -> tuple[dict[str, Any], bool] | None:
    cleaned = _SPECIAL_TOKEN_RE.sub("", str(blob or "")).strip()
    if not cleaned:
        return None
    jm = _JSON_RE.search(cleaned)
    if not jm:
        return None
    try:
        obj = json.loads(jm.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not name:
        return None
    name = str(name).lower()
    if name not in _HARNESS_G_TOOL_NAMES:
        return None
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}, True


def parse_harness_g_action(
    text: str,
    *,
    action_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool]:
    blob = str(text or "").strip()
    if not blob:
        return {"name": "unknown", "arguments": {}}, False
    parsed_json = _parse_json_harness_g_action(blob)
    if parsed_json is not None:
        name = str(parsed_json[0].get("name") or "")
        args = dict(parsed_json[0].get("arguments") or {})
        if name in {"select", "lookup", "answer_with"}:
            key = "sid" if name != "lookup" else "eid"
            if not str(args.get(key) or "").strip():
                return {"name": "unknown", "arguments": {}}, False
        return parsed_json
    if "<|call|>" in blob:
        try:
            parsed = parse_codec_action(blob)
            name = str(parsed.get("name") or "").lower()
            args = dict(parsed.get("arguments") or {})
            if name in _HARNESS_G_TOOL_NAMES:
                if name in {"select", "lookup", "answer_with"}:
                    key = "sid" if name != "lookup" else "eid"
                    if not str(args.get(key) or "").strip():
                        return {"name": "unknown", "arguments": {}}, False
                return {"name": name, "arguments": args}, True
        except Exception:
            pass
    match = _TO_RE.search(blob)
    if match and "<|call|>" in blob:
        name = match.group("name").lower()
        args: dict[str, Any] | None = None
        jm = _JSON_RE.search(blob[match.end() :])
        if jm:
            try:
                loaded = json.loads(jm.group(0))
                if isinstance(loaded, dict):
                    args = loaded
            except json.JSONDecodeError:
                args = None
        if name in {"select", "lookup", "answer_with"}:
            key = "sid" if name != "lookup" else "eid"
            if not args or not str(args.get(key) or "").strip():
                return {"name": "unknown", "arguments": {}}, False
        if args is None and name not in {"init", "answer"}:
            return {"name": "unknown", "arguments": {}}, False
        return {"name": name, "arguments": args or {}}, True
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
    actions_obs: list[tuple[Any, Any]] | None = None,
) -> list[int]:
    include_aw = _include_answer_with(harness_mask)
    if enc is None:
        return []
    tokenizer = getattr(enc, "tokenizer", None)
    if _is_qwen_family(enc) and tokenizer is not None:
        messages = _qwen_messages(
            query,
            wm_text,
            actions_obs,
            include_answer_with=include_aw,
        )
        kwargs: dict[str, Any] = {
            "tools": qwen_harness_g_tool_schemas(include_answer_with=include_aw),
            "add_generation_prompt": True,
            "tokenize": True,
        }
        try:
            raw = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            raw = tokenizer.apply_chat_template(messages, **kwargs)
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

    conv = build_harness_g_context(
        query,
        wm_text,
        actions_obs,
        include_answer_with=include_aw,
    )
    try:
        ids = harmony.render_conversation_for_completion(conv, Role.ASSISTANT)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Harness-G Harmony renderer failed: {type(exc).__name__}: {exc}"
        ) from exc
    return assert_o200k_harmony_token_ids(ids, what="Harness-G Harmony prompt")
