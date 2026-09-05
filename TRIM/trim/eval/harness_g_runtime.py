"""Harness-G prompt + action parse (named tools, not ephemeral A0 ids).

Policy interface inside TRIM is Harmony-like ``to=select {json}``. The
upstream Harness-G A0 menu is mapped onto select / lookup / answer /
answer_with so OPD can score a stable action name.
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


def render_prompt(query: str, wm_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + f"\nQuestion: {query}\n"
        + (wm_text or "")
        + "\nEmit one tool call: to=select|lookup|answer {json}\n"
    )


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


def build_prompt_ids(query: str, wm_text: str, enc) -> list[int]:
    prompt = render_prompt(query, wm_text)
    if enc is None:
        return []
    family = str(getattr(enc, "family", "") or "")
    tokenizer = getattr(enc, "tokenizer", None)
    if family == "qwen3" and tokenizer is not None:
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
    encode = getattr(enc, "encode", None)
    if encode is None:
        return []
    try:
        return [int(x) for x in encode(prompt)]
    except Exception:
        return []
