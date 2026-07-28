"""Harmony / tool-call ↔ CapabilityAction adapters."""

from __future__ import annotations

import json
import re
from typing import Any

from harness.capability.action_space import CapabilityAction, CapabilityActionType

_TOOL_NAME_MAP = {
    "search_corpus": CapabilityActionType.SEARCH,
    "fan_out_search": CapabilityActionType.SEARCH,
    "grep_corpus": CapabilityActionType.GREP,
    "read_document": CapabilityActionType.OPEN_DOCUMENT,
    "curate": CapabilityActionType.CURATE_DOCUMENT,
    "review_docs": CapabilityActionType.REVIEW_DOCS,
    "verify": CapabilityActionType.VERIFY_CLAIM,
    "end_search": CapabilityActionType.STOP_AND_ANSWER,
    "user_text": CapabilityActionType.ANSWER,
    "answer": CapabilityActionType.ANSWER,
}

_ACTION_TO_TOOL = {
    CapabilityActionType.SEARCH: "search_corpus",
    CapabilityActionType.REWRITE_QUERY: "search_corpus",
    CapabilityActionType.CONTINUE_SEARCH: "search_corpus",
    CapabilityActionType.GREP: "grep_corpus",
    CapabilityActionType.OPEN_DOCUMENT: "read_document",
    CapabilityActionType.CURATE_DOCUMENT: "curate",
    CapabilityActionType.UPDATE_EVIDENCE: "curate",
    CapabilityActionType.REVIEW_DOCS: "review_docs",
    CapabilityActionType.VERIFY_CLAIM: "verify",
    CapabilityActionType.STOP_AND_ANSWER: "end_search",
    CapabilityActionType.ANSWER: "user_text",
    CapabilityActionType.ABSTAIN: "end_search",
}


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    # Try whole-string JSON
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                objs.append(parsed)
            elif isinstance(parsed, list):
                objs.extend(x for x in parsed if isinstance(x, dict))
            return objs
        except json.JSONDecodeError:
            pass
    # Fallback: find tool-like dicts
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                objs.append(parsed)
        except json.JSONDecodeError:
            continue
    return objs


def parse_policy_action(raw_action: str) -> CapabilityAction | None:
    """Parse a raw policy/tool string into CapabilityAction. Returns None on failure."""
    if not raw_action or not str(raw_action).strip():
        return None

    text = str(raw_action).strip()

    # Harmony-ish: tool name followed by JSON params
    for tool_name, action_type in _TOOL_NAME_MAP.items():
        if tool_name in text:
            objs = _extract_json_objects(text)
            args: dict[str, Any] = {}
            for obj in objs:
                if "tool" in obj and obj.get("tool") != tool_name and "name" not in obj:
                    continue
                # Prefer objects that look like params for this tool
                if obj.get("tool") == tool_name or obj.get("name") == tool_name:
                    args = {k: v for k, v in obj.items() if k not in {"tool", "name"}}
                    break
                if any(k in obj for k in ("query", "queries", "doc_ids", "add_ids", "claim", "pattern", "doc_id")):
                    args = dict(obj)
                    break
            if action_type == CapabilityActionType.SEARCH and "queries" in args:
                return CapabilityAction(
                    action_type=action_type,
                    arguments={"queries": args.get("queries"), "fan_out": True},
                )
            claim_id = None
            if action_type == CapabilityActionType.VERIFY_CLAIM:
                claim_id = str(args.get("claim", ""))[:64] or None
            return CapabilityAction(
                action_type=action_type,
                arguments=args,
                target_claim_id=claim_id,
            )

    # CapabilityAction serialized form
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            return CapabilityAction.from_dict(data)
    except json.JSONDecodeError:
        pass

    return None


def parse_action_from_tools(
    tool_names: list[str],
    params_list: list[dict[str, Any]],
) -> CapabilityAction | None:
    """Parse from already-decoded tool names + params (preferred path)."""
    if not tool_names:
        return None
    name = tool_names[0]
    params = params_list[0] if params_list else {}
    action_type = _TOOL_NAME_MAP.get(name)
    if action_type is None:
        return CapabilityAction(
            action_type=CapabilityActionType.UNKNOWN,
            arguments={"tool": name, **dict(params)},
        )
    args = dict(params)
    if name == "fan_out_search":
        args = {"queries": params.get("queries", []), "fan_out": True}
    claim_id = None
    if action_type == CapabilityActionType.VERIFY_CLAIM:
        claim_id = str(params.get("claim", ""))[:64] or None
    return CapabilityAction(
        action_type=action_type,
        arguments=args,
        target_claim_id=claim_id,
    )


def render_capability_action(action: CapabilityAction) -> str:
    """Render CapabilityAction as a Harmony-style tool call JSON string."""
    tool = _ACTION_TO_TOOL.get(action.action_type, "search_corpus")
    args = dict(action.arguments)

    if action.action_type == CapabilityActionType.REWRITE_QUERY:
        query = args.get("query") or args.get("target_claim") or ""
        payload = {"tool": tool, "query": query}
    elif action.action_type == CapabilityActionType.CONTINUE_SEARCH:
        payload = {"tool": tool, "query": args.get("query", "")}
    elif action.action_type == CapabilityActionType.SEARCH and args.get("fan_out"):
        payload = {"tool": "fan_out_search", "queries": args.get("queries", [])}
    elif action.action_type == CapabilityActionType.STOP_AND_ANSWER:
        payload = {"tool": "end_search", "reasoning": args.get("reasoning", "evidence sufficient")}
    elif action.action_type == CapabilityActionType.ANSWER:
        payload = {
            "tool": "user_text",
            "text": args.get("text") or args.get("answer") or args.get("reasoning", ""),
        }
    elif action.action_type == CapabilityActionType.ABSTAIN:
        payload = {"tool": "end_search", "reasoning": args.get("reasoning", "abstain")}
    elif action.action_type == CapabilityActionType.UPDATE_EVIDENCE:
        payload = {
            "tool": "curate",
            "add_ids": args.get("add_ids", []),
            "remove_ids": args.get("remove_ids", []),
            "status": args.get("status"),
        }
    else:
        payload = {"tool": tool, **args}

    return json.dumps(payload, ensure_ascii=False)
