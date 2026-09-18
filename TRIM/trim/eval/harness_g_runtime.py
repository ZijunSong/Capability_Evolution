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
_AID_FULL_RE = re.compile(r"^\s*(A\d+)\s*$", re.I)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_HARMONY_CALL_RE = re.compile(
    r"to=functions\.(?P<name>init|select|lookup|answer|answer_with)\b"
    r"(?P<body>.*?)(?=<\|call\|>)",
    re.I | re.DOTALL,
)
_ASSISTANT_START_RE = re.compile(r"<\|start\|>assistant")
_METADATA_ONLY_RE = re.compile(
    r"^(title|author|date|published|copyright|table of contents|references):\s*.+\s*$",
    re.I,
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")
PROTOCOL_FEEDBACK_KEY = "__protocol_feedback__"

SYSTEM_PROMPT = """You are a Harness-G search agent.
Basic runtime tools (always available):
- init: first retrieve visible evidence sentences
- select: commit a currently listed SELECT sid as evidence
- lookup: follow a currently listed LOOKUP eid; the environment builds the retrieval query
- answer: stop using already selected evidence. Only legal when ANSWER appears under actions.
Optional when listed under actions:
- answer_with: atomically SELECT one currently unselected visible sentence and stop. If evidence is already selected, call answer instead.
Rules:
- Execute ONLY the actions and targets listed under the actions section of working memory.
- Selected ids are known evidence; they are not SELECT/ANSWER_WITH targets unless they reappear there.
- Never invent free-form search queries or new ids. Never substitute surface names for eid.
- Every turn emit exactly ONE executable tool call with a complete JSON object.
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
        "description": "Commit a currently listed SELECT sid as evidence. The sid must appear under actions as SELECT.",
        "parameters": {
            "type": "object",
            "properties": {
                "sid": {"type": "string", "description": "Unselected visible sentence id from the current SELECT menu."}
            },
            "required": ["sid"],
        },
    },
    {
        "name": "lookup",
        "description": "Follow a currently listed LOOKUP eid; the environment builds the retrieval query.",
        "parameters": {
            "type": "object",
            "properties": {
                "eid": {"type": "string", "description": "Entity id from the current LOOKUP menu."}
            },
            "required": ["eid"],
        },
    },
    {
        "name": "answer",
        "description": "Stop using already selected evidence. Legal only when ANSWER is listed under actions.",
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
    "description": (
        "Atomically SELECT one currently unselected visible sentence from the menu and stop. "
        "Do not pass an already selected sid; if evidence is already selected, call answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sid": {
                "type": "string",
                "description": "Unselected visible sentence id listed as ANSWER_WITH in the current menu.",
            }
        },
        "required": ["sid"],
    },
}

_ALLOWED_ARGS: dict[str, frozenset[str]] = {
    "init": frozenset(),
    "select": frozenset({"sid"}),
    "lookup": frozenset({"eid"}),
    "answer": frozenset({"reason", "reasoning"}),
    "answer_with": frozenset({"sid", "sids"}),
}

REASONING_EFFORTS = ("low", "medium", "high")
DEFAULT_REASONING_EFFORT = "high"


def render_prompt(query: str, wm_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + f"\nQuestion: {query}\n"
        + (wm_text or "")
        + "\nEmit one tool call using init, select, lookup, or answer.\n"
    )


def _is_hf_chat_family(enc: Any) -> bool:
    from trim.eval.model_profiles import is_hf_chat_family

    return is_hf_chat_family(getattr(enc, "family", ""))


def _is_qwen_family(enc: Any) -> bool:
    """Backward-compatible alias."""
    return _is_hf_chat_family(enc)


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


def is_protocol_feedback(action: Any) -> bool:
    return isinstance(action, dict) and bool(action.get(PROTOCOL_FEEDBACK_KEY))


def make_protocol_feedback(message: str) -> dict[str, Any]:
    return {PROTOCOL_FEEDBACK_KEY: True, "content": str(message or "")}


def _action_name_args(action: Any) -> tuple[str, dict[str, Any]]:
    if is_protocol_feedback(action):
        return "", {}
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
    from trim.eval.harmony_runtime import prompt_history_keep, recent_actions_obs

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}"},
    ]
    for action, obs in recent_actions_obs(list(actions_obs or []), keep=prompt_history_keep()):
        if is_protocol_feedback(action):
            messages.append({"role": "user", "content": _obs_text(obs)})
            continue
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
    if wm_text:
        messages.append({"role": "user", "content": str(wm_text)})
    if actions_obs:
        messages.append(
            {
                "role": "user",
                "content": "Continue the Harness-G search. Emit exactly one tool call.",
            }
        )
    return messages


def resolve_reasoning_effort(value: str | None) -> Any:
    from openai_harmony import ReasoningEffort

    key = str(value or DEFAULT_REASONING_EFFORT).strip().lower()
    mapping = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }
    return mapping.get(key, ReasoningEffort.HIGH)


def build_harness_g_conversation(
    query: str,
    wm_text: str,
    *,
    include_answer_with: bool = False,
    reasoning_effort: str | None = None,
) -> Any:
    """Harmony Conversation with Harness-G function tools (not Harness-1 tools)."""
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
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
        .with_reasoning_effort(resolve_reasoning_effort(reasoning_effort))
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
    reasoning_effort: str | None = None,
) -> Any:
    """Harmony context with Harness-G tools and recent tool-call history."""
    from openai_harmony import Conversation, DeveloperContent, Message, Role, SystemContent, ToolDescription
    from trim.eval.harmony_runtime import _ensure_scope, make_action, make_observation, prompt_history_keep, recent_actions_obs

    _ensure_scope()
    from harness.ultra_core import action_observation_to_messages

    tools = [
        ToolDescription.new(str(spec["name"]), str(spec["description"]), spec["parameters"])
        for spec in _tool_specs(include_answer_with=include_answer_with)
    ]
    system = (
        SystemContent.new()
        .with_reasoning_effort(resolve_reasoning_effort(reasoning_effort))
        .with_conversation_start_date("2026-04-01")
    )
    developer = (
        DeveloperContent.new()
        .with_instructions(SYSTEM_PROMPT.strip())
        .with_function_tools(tools)
    )
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.DEVELOPER, developer),
        Message.from_role_and_content(Role.USER, f"Question: {query}"),
    ]
    for action, obs in recent_actions_obs(list(actions_obs or []), keep=prompt_history_keep()):
        if is_protocol_feedback(action):
            messages.append(Message.from_role_and_content(Role.USER, _obs_text(obs)))
            continue
        if hasattr(action, "tools"):
            act_obj = action
            obs_obj = obs
        else:
            name, args = _action_name_args(action)
            act_obj = make_action(name, args)
            obs_obj = obs if hasattr(obs, "observations") else make_observation(_obs_text(obs))
        messages.extend(action_observation_to_messages(act_obj, obs_obj, compress=False))
    if wm_text:
        messages.append(Message.from_role_and_content(Role.USER, str(wm_text)))
    if actions_obs:
        messages.append(
            Message.from_role_and_content(
                Role.USER,
                "Continue the Harness-G search. Emit exactly one tool call using the registered functions.",
            )
        )
    return Conversation(messages=messages)


def _validate_args(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    allowed = _ALLOWED_ARGS.get(name)
    if allowed is None:
        return None
    extra = set(args) - set(allowed)
    if extra:
        return None
    for key, value in args.items():
        if key == "sids":
            if not isinstance(value, list) or not all(isinstance(x, (str, int)) for x in value):
                return None
            continue
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            return None
    if name in {"select", "answer_with"}:
        sid = str(args.get("sid") or "")
        if not sid.strip():
            sids = args.get("sids") or []
            if not sids or not str(sids[0]).strip():
                return None
            args = dict(args)
            args["sid"] = str(sids[0])
    if name == "lookup" and not str(args.get("eid") or "").strip():
        return None
    return dict(args)


def _action_from_name_args(name: str, args: dict[str, Any] | None) -> tuple[dict[str, Any], bool] | None:
    name = str(name or "").lower()
    if name not in _HARNESS_G_TOOL_NAMES:
        return None
    if args is None:
        return None
    checked = _validate_args(name, args)
    if checked is None:
        return None
    return {"name": name, "arguments": checked}, True


def _action_from_json_obj(obj: dict[str, Any]) -> tuple[dict[str, Any], bool] | None:
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not name:
        return None
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return _action_from_name_args(str(name).lower(), args)


def _loads_complete_json_object(text: str) -> dict[str, Any] | None:
    blob = str(text or "").strip()
    if not blob:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_json_harness_g_action(blob: str) -> tuple[dict[str, Any], bool] | None:
    cleaned = _SPECIAL_TOKEN_RE.sub("", str(blob or "")).strip()
    if not cleaned:
        return None
    obj = _loads_complete_json_object(cleaned)
    if obj is None:
        return None
    return _action_from_json_obj(obj)


def _iter_harmony_messages(blob: str) -> list[dict[str, Any]]:
    """Parse Harmony messages by start/end token boundaries.

    Nested control tokens inside an open message body are treated as text.
    Analysis messages end only at <|end|>; tool messages end at call/return/end.
    """
    messages: list[dict[str, Any]] = []
    pos = 0
    start_tag = "<|start|>"
    while True:
        start = blob.find(start_tag, pos)
        if start < 0:
            break
        rest = blob[start + len(start_tag) :]
        header_end = rest.find("<|message|>")
        if header_end < 0:
            header = rest
            body = ""
            after_header = rest
            msg_start_in_rest = len(rest)
        else:
            header = rest[:header_end]
            after_header = rest[header_end + len("<|message|>") :]
            msg_start_in_rest = header_end + len("<|message|>")
        header_l = header.lower()
        role = "assistant" if header_l.startswith("assistant") else header.split("<|", 1)[0].strip().split()[0] if header.strip() else ""
        recipient = ""
        m_to = re.search(r"to=(functions\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)", header, re.I)
        if m_to:
            recipient = m_to.group(1)
        channel = ""
        m_ch = re.search(r"<\|channel\|>([A-Za-z0-9_]+)", header, re.I)
        if m_ch:
            channel = m_ch.group(1).lower()
        if channel == "analysis" or (not recipient and channel != "commentary"):
            end_tokens = ("<|end|>",)
        else:
            end_tokens = ("<|call|>", "<|return|>", "<|end|>")
        end_pos = -1
        end_tok = ""
        body = after_header if header_end >= 0 else ""
        search_from = 0
        while True:
            found = [(body.find(tok, search_from), tok) for tok in end_tokens]
            found = [(i, tok) for i, tok in found if i >= 0]
            if not found:
                break
            i, tok = min(found, key=lambda x: x[0])
            end_pos = i
            end_tok = tok
            break
        if end_pos < 0:
            payload = body
            consumed = len(rest)
        else:
            payload = body[:end_pos] if header_end >= 0 else ""
            consumed = (msg_start_in_rest if header_end >= 0 else len(header)) + end_pos + len(end_tok)
        messages.append(
            {
                "role": role.lower(),
                "recipient": recipient,
                "channel": channel,
                "body": payload,
                "end_token": end_tok or None,
            }
        )
        pos = start + len(start_tag) + consumed
    return messages


def _harmony_executable_calls(blob: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for msg in _iter_harmony_messages(blob):
        if msg.get("role") != "assistant":
            continue
        if msg.get("channel") == "analysis":
            continue
        recipient = str(msg.get("recipient") or "")
        name = recipient.split(".", 1)[-1].lower() if recipient else ""
        if name not in _HARNESS_G_TOOL_NAMES:
            continue
        if msg.get("end_token") not in {"<|call|>", "<|return|>"}:
            continue
        calls.append(msg)
    return calls


def _harmony_executable_region(blob: str) -> str | None:
    calls = _harmony_executable_calls(blob)
    if len(calls) != 1:
        return None
    return json.dumps(calls[0], ensure_ascii=False)


def _parse_harmony_call_region(region: str) -> tuple[dict[str, Any], bool] | None:
    try:
        msg = json.loads(region)
    except json.JSONDecodeError:
        msg = None
    if not isinstance(msg, dict):
        calls = _harmony_executable_calls(region)
        if len(calls) != 1:
            return None
        msg = calls[0]
    recipient = str(msg.get("recipient") or "")
    name = recipient.split(".", 1)[-1].lower() if recipient else ""
    args = _loads_complete_json_object(str(msg.get("body") or ""))
    if args is None:
        return None
    parsed = _action_from_name_args(name, args)
    return parsed


def parse_harness_g_action(
    text: str,
    *,
    action_map: Mapping[str, Mapping[str, Any]] | None = None,
    strict: bool = True,
    finish_reason: str | None = None,
) -> tuple[dict[str, Any], bool]:
    del strict
    blob = str(text or "").strip()
    if not blob:
        return {"name": "unknown", "arguments": {}}, False

    if str(finish_reason or "") == "length":
        return {"name": "truncated", "arguments": {"finish_reason": "length"}}, False

    tool_calls = list(_TOOL_CALL_RE.finditer(blob))
    if len(tool_calls) > 1:
        return {"name": "unknown", "arguments": {}}, False
    if len(tool_calls) == 1:
        obj = _loads_complete_json_object(tool_calls[0].group(1))
        if isinstance(obj, dict):
            parsed = _action_from_json_obj(obj)
            if parsed is not None:
                return parsed
        return {"name": "unknown", "arguments": {}}, False

    if "<|start|>" in blob or "<|call|>" in blob or "<|return|>" in blob:
        calls = _harmony_executable_calls(blob)
        if len(calls) != 1:
            return {"name": "unknown", "arguments": {}}, False
        parsed_region = _parse_harmony_call_region(json.dumps(calls[0], ensure_ascii=False))
        if parsed_region is not None:
            return parsed_region
        return {"name": "unknown", "arguments": {}}, False

    aid_match = _AID_FULL_RE.match(blob)
    if aid_match and action_map:
        mapped = action_map.get(aid_match.group(1)) or action_map.get(aid_match.group(1).upper())
        if mapped:
            name = str(mapped.get("name") or mapped.get("type") or "").lower()
            args: dict[str, Any] = {}
            if mapped.get("sid"):
                args["sid"] = mapped["sid"]
            if mapped.get("eid"):
                args["eid"] = mapped["eid"]
            if mapped.get("sids"):
                args["sids"] = list(mapped["sids"])
            parsed = _action_from_name_args(name, args)
            if parsed is not None:
                return parsed

    bare = re.fullmatch(
        r"to=(?:functions\.)?(init|select|lookup|answer|answer_with)\s+(\{.*\})\s*",
        blob,
        re.I | re.DOTALL,
    )
    if bare:
        args = _loads_complete_json_object(bare.group(2))
        parsed = _action_from_name_args(bare.group(1).lower(), args or None)
        if parsed is not None:
            return parsed

    return {"name": "unknown", "arguments": {}}, False


def build_prompt_ids(
    query: str,
    wm_text: str,
    enc,
    *,
    harness_mask: Mapping[str, Any] | None = None,
    actions_obs: list[tuple[Any, Any]] | None = None,
    reasoning_effort: str | None = None,
) -> list[int]:
    include_aw = _include_answer_with(harness_mask)
    if enc is None:
        return []
    tokenizer = getattr(enc, "tokenizer", None)
    if _is_hf_chat_family(enc) and tokenizer is not None:
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
        from trim.eval.model_profiles import resolve_model_profile
        from trim.eval.model_tokenizer import _to_token_ids, assert_hf_chat_prompt_ids

        profile = resolve_model_profile(getattr(enc, "source", "") or "", tokenizer)
        return assert_hf_chat_prompt_ids(
            _to_token_ids(raw),
            what=f"Harness-G {profile.label} prompt",
            strict_qwen=profile.strict_qwen_prompt_ids,
        )
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
        reasoning_effort=reasoning_effort,
    )
    try:
        ids = harmony.render_conversation_for_completion(conv, Role.ASSISTANT)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Harness-G Harmony renderer failed: {type(exc).__name__}: {exc}"
        ) from exc
    return assert_o200k_harmony_token_ids(ids, what="Harness-G Harmony prompt")
