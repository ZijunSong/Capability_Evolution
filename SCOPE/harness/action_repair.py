"""Repair / normalize weak-policy tool calls for Ultra harness chat drivers.

Addresses failure modes observed on BrowseComp full harness rollouts with
Qwen2.5-Instruct:
  - inventing a `Document` tool (legacy subagent final-answer format)
  - emitting `<Document id=...>` as plain text (treated as episode end)
  - malformed JSON / missing required keys
  - premature `end_search` with empty curated set
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import json_repair

from harness.tasks import FINAL_OUTPUT_DOCUMENT_PATTERN
from harness.tools import Tool, ToolSet, UserTextTool
from harness.trajectory import Action

_DOC_TOOL_ALIASES = {
    "document",
    "documents",
    "Document",
    "Documents",
    "final_answer",
    "submit_documents",
    "submit",
}

_ID_KEY_CANDIDATES = (
    "id",
    "doc_id",
    "document_id",
    "document_ids",
    "doc_ids",
    "ids",
    "add_ids",
)


def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_as_id_list(item))
        return out
    if isinstance(value, dict):
        for key in _ID_KEY_CANDIDATES:
            if key in value:
                return _as_id_list(value[key])
        return []
    return [str(value)]


def extract_document_ids_from_text(text: str) -> list[str]:
    ids = [m.group("chunk_id") for m in FINAL_OUTPUT_DOCUMENT_PATTERN.finditer(text or "")]
    # Also accept bare "DOCUMENT ID: xxx" lines
    ids.extend(re.findall(r"(?i)document\s*id\s*[:=]\s*([A-Za-z0-9_\-:./]+)", text or ""))
    # Dedup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _safe_parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = json_repair.loads(text)
        except Exception:  # noqa: BLE001
            return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        # fan_out style: ["q1", "q2"] or [{...}]
        if parsed and all(isinstance(x, str) for x in parsed):
            return {"queries": parsed}
        if parsed and isinstance(parsed[0], dict):
            return dict(parsed[0])
    if isinstance(parsed, str):
        return {"query": parsed}
    return {}


def normalize_tool_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Coerce common weak-model argument shapes into schema-compatible dicts."""
    p = dict(params or {})
    if name in {"search_corpus", "grep_corpus"}:
        if name == "search_corpus" and "query" not in p:
            if "q" in p:
                p["query"] = p.pop("q")
            elif "queries" in p:
                qs = p.get("queries")
                if isinstance(qs, list) and qs:
                    p["query"] = qs[0]
                elif isinstance(qs, str):
                    p["query"] = qs
            elif "pattern" in p:
                p["query"] = p["pattern"]
        if name == "grep_corpus" and "pattern" not in p:
            if "query" in p:
                p["pattern"] = p["query"]
            elif "q" in p:
                p["pattern"] = p["q"]
        # Unwrap accidental nesting: {"query": {"query": "..."}}
        for key in ("query", "pattern"):
            if isinstance(p.get(key), dict):
                inner = p[key]
                p[key] = inner.get(key) or inner.get("q") or inner.get("pattern") or str(inner)
    elif name == "fan_out_search":
        if "queries" not in p:
            if "query" in p:
                q = p["query"]
                p["queries"] = q if isinstance(q, list) else [q]
            elif "q" in p:
                p["queries"] = [p["q"]]
        if isinstance(p.get("queries"), str):
            p["queries"] = [p["queries"]]
    elif name == "curate":
        if "add_ids" not in p:
            for key in _ID_KEY_CANDIDATES:
                if key in p and key != "add_ids":
                    p["add_ids"] = _as_id_list(p[key])
                    break
        else:
            p["add_ids"] = _as_id_list(p.get("add_ids"))
        if "remove_ids" in p:
            p["remove_ids"] = _as_id_list(p.get("remove_ids"))
        else:
            p.setdefault("remove_ids", [])
    elif name == "read_document":
        if "doc_id" not in p and "id" not in p:
            ids = _as_id_list(p)
            if ids:
                p["doc_id"] = ids[0]
    elif name == "review_docs" or name == "verify":
        if "doc_ids" not in p:
            p["doc_ids"] = _as_id_list(p.get("ids") or p.get("add_ids") or p.get("doc_id"))
        else:
            p["doc_ids"] = _as_id_list(p.get("doc_ids"))
    elif name == "end_search":
        if "reasoning" not in p:
            p["reasoning"] = str(p.get("reason") or p.get("text") or "done")[:500]
    return p


def _lookup_tool(toolset: ToolSet, name: str) -> Optional[Tool]:
    tool = toolset.get_tool(name)
    if tool is not None:
        return tool
    # case-insensitive fallback
    lower = name.lower()
    for tname, tool in toolset.tools.items():
        if tname.lower() == lower:
            return tool
    return None


def repair_action_from_tool_calls(
    *,
    tool_calls: list[tuple[str, Any]],
    toolset: ToolSet,
    curate_tool: Tool,
    end_search_tool: Tool,
) -> Action:
    """Build an Action from raw (name, args) pairs with alias / param repair."""
    tools: list[Tool] = []
    params_list: list[dict[str, Any]] = []
    sources: list[str] = []

    pending_doc_ids: list[str] = []

    for idx, (raw_name, raw_args) in enumerate(tool_calls):
        name = str(raw_name or "").strip()
        args = _safe_parse_args(raw_args)

        if name in _DOC_TOOL_ALIASES or name.lower() in {a.lower() for a in _DOC_TOOL_ALIASES}:
            pending_doc_ids.extend(_as_id_list(args))
            pending_doc_ids.extend(extract_document_ids_from_text(json.dumps(args)))
            continue

        tool = _lookup_tool(toolset, name)
        if tool is None:
            # Unknown tool — try to salvage document ids from args/name
            pending_doc_ids.extend(_as_id_list(args))
            continue

        norm = normalize_tool_params(tool.tool_schema.name, args)
        # Drop clearly invalid required-key calls (will be replaced below if empty)
        tname = tool.tool_schema.name
        if tname == "search_corpus" and not norm.get("query"):
            continue
        if tname == "grep_corpus" and not norm.get("pattern"):
            continue
        if tname == "fan_out_search" and not norm.get("queries"):
            continue
        if tname == "curate" and not norm.get("add_ids") and not norm.get("remove_ids"):
            continue
        if tname == "read_document" and not (norm.get("doc_id") or norm.get("id")):
            continue

        tools.append(tool)
        params_list.append(norm)
        sources.append(f"call_{idx}")

    if pending_doc_ids:
        # Prefer curate when we still have salvageable IDs; caller may end later.
        tools.append(curate_tool)
        params_list.append({"add_ids": pending_doc_ids[:30], "remove_ids": []})
        sources.append("repaired_document")

    if not tools:
        # Last resort: do not invent end_search here — caller decides.
        return Action(tools=[], params=[], sources=[])

    return Action(tools=tools, params=params_list, sources=sources)


def maybe_convert_user_text_to_tools(
    action: Action,
    *,
    curate_tool: Tool,
    end_search_tool: Tool,
    allow_end: bool,
) -> Action:
    """If the model emitted Document XML / free text, convert to curate(+optional end)."""
    if not action.tools:
        return action
    if not any(isinstance(t, UserTextTool) for t in action.tools):
        return action

    texts = []
    for tool, params in zip(action.tools, action.params):
        if isinstance(tool, UserTextTool):
            texts.append(str((params or {}).get("text", "")))
    blob = "\n".join(texts)
    doc_ids = extract_document_ids_from_text(blob)
    if not doc_ids:
        # Plain text stop without Document tags — keep as-is (env will end).
        return action

    tools: list[Tool] = [curate_tool]
    params_list: list[dict[str, Any]] = [{"add_ids": doc_ids[:30], "remove_ids": []}]
    sources = ["repaired_user_text"]
    if allow_end:
        tools.append(end_search_tool)
        params_list.append({"reasoning": "submitted Document tags"})
        sources.append("repaired_user_text_end")
    return Action(tools=tools, params=params_list, sources=sources)


def should_block_early_end(
    *,
    turn: int,
    n_curated: int,
    n_pool: int,
    min_turns: int,
    min_curated: int,
) -> tuple[bool, str]:
    """Return (block, reason) for premature end_search / user_text stop."""
    if turn < min_turns:
        return True, (
            f"Too early to end (turn {turn}/{min_turns}). Continue the search→curate "
            "loop: try a different query angle, then curate relevant docs."
        )
    if n_curated < min_curated and n_pool > 0:
        return True, (
            f"Curated set is empty/too small ({n_curated}) but pool has {n_pool} docs. "
            "Call curate NOW to add ALL plausibly relevant pool docs before ending."
        )
    if n_pool == 0 and turn < max(min_turns, 4):
        return True, (
            "No documents in pool yet. Call fan_out_search or search_corpus before ending."
        )
    return False, ""
