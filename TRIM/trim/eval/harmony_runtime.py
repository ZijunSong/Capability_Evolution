"""Canonical Harmony tool-call runtime for clean gpt-oss / Harness-1.

Matches the public-SFT conversion path:
  build_context() + openai_harmony.render_conversation_for_completion
Generation MUST stop on assistant-action tokens (<|call|>, <|return|>),
never on <|end|> (that would cut off analysis before the tool call).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

CANONICAL_TOOLS = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
)
SCHEMA_TOOLS = CANONICAL_TOOLS + ("multi_tool_use",)

SCOPE = Path("/data/ppnm/Capability_Evolution/SCOPE")
REPO = Path(__file__).resolve().parents[2]
_HARNESS_CANDIDATES = (
    REPO / "external" / "harness-1",
    REPO.parent / "SCAPE" / "external" / "harness-1",
)
LOCAL_HARNESS = next((p for p in _HARNESS_CANDIDATES if p.is_dir()), _HARNESS_CANDIDATES[0])

# Strict: recipient must be an identifier, not a prose blob.
_TO_RE = re.compile(
    r"(?:to=|recipient[\"']?\s*[:=]\s*[\"']?)(?:functions\.)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
_MESSAGE_JSON_RE = re.compile(
    r"<\|message\|>(?P<body>\{.*\})<\|call\|>",
    re.DOTALL,
)
_JSON_AFTER_TO_RE = re.compile(
    r"to=(?:functions\.)?[A-Za-z_][A-Za-z0-9_]*[^\{]{0,200}(?P<body>\{.*\})",
    re.DOTALL,
)
_CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CHANNEL_LEAK_RE = re.compile(
    r"(?:commentary|analysis|channel|constrain|json|assistant|functions|"
    r"message|call|start|end|return)+",
    re.I,
)
_TRAILING_CHANNEL_RE = re.compile(
    r"(?:commentary|analysis|channel|constrain|json|assistant|functions|"
    r"message|call|start|end|return)+$",
    re.I,
)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _schema_tool_from_token(token: str) -> str | None:
    if not token:
        return None
    if token in SCHEMA_TOOLS:
        return token
    lowered = token.lower()
    if lowered in SCHEMA_TOOLS:
        return lowered
    for tool in sorted(SCHEMA_TOOLS, key=len, reverse=True):
        if not lowered.startswith(tool.lower()):
            continue
        rest_norm = re.sub(r"[^A-Za-z0-9]+", "", token[len(tool) :]).lower()
        if rest_norm == "" or _CHANNEL_LEAK_RE.fullmatch(rest_norm):
            return tool
    hits = [tool for tool in SCHEMA_TOOLS if _edit_distance(lowered, tool) <= 1]
    if len(hits) == 1:
        return hits[0]
    return None


def _canonicalize_tool_name(name: str | None) -> str | None:
    """Recover a schema tool when Harmony channel tokens leak into the recipient.

    gpt-oss often emits `to=functions.curate?commentary` instead of
    `to=functions.curate<|channel|>commentary`. Keep true unknown names intact.
    """
    if name is None:
        return None
    raw = str(name).strip()
    if not raw:
        return None
    raw = _CTRL_TOKEN_RE.sub("", raw).replace("functions.", "").strip()
    ident_m = _IDENT_RE.match(raw)
    ident = ident_m.group(0) if ident_m else ""
    for token in (raw, ident, _TRAILING_CHANNEL_RE.sub("", ident or raw)):
        hit = _schema_tool_from_token(token)
        if hit:
            return hit
    if ident in ("functions", "None", "none"):
        return None
    return ident or raw


def _ensure_scope() -> None:
    # Prefer this repository's pinned Harness-1. The historical /data path may
    # not exist (or may contain a different Harness checkout) on formal hosts.
    if LOCAL_HARNESS.is_dir() and str(LOCAL_HARNESS) not in sys.path:
        sys.path.insert(0, str(LOCAL_HARNESS))
    if SCOPE.is_dir() and str(SCOPE) not in sys.path:
        sys.path.append(str(SCOPE))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


# gpt-oss / Harness-1 Harmony. Never cl100k_base.
O200K_HARMONY = "o200k_harmony"
CANONICAL_STOP_TOKEN_IDS = [200012, 200002]  # <|call|>, <|return|> — not <|end|>


def load_harmony_enc():
    _ensure_scope()
    # Prefer the approved offline bundle when callers did not export paths.
    bundle = Path("/opt/scape-projected-action/share/tiktoken-bundle")
    if bundle.is_dir():
        os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", str(bundle / "share/tiktoken"))
        os.environ.setdefault("TIKTOKEN_RS_CACHE_DIR", str(bundle / "tiktoken_rs_cache"))
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(bundle / "tiktoken_cache"))
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except Exception:
        # The GPT-OSS tiktoken vocabulary is not always reachable in the
        # approved offline runtime. Use the repository's deterministic local
        # Harmony-compatible fallback rather than silently switching models.
        from harness._local_harmony_fallback import _LocalHarmonyEncodingFallback

        enc = _LocalHarmonyEncodingFallback()
    stops = [int(x) for x in enc.stop_tokens_for_assistant_actions()]
    if 200012 not in stops or 200002 not in stops:
        raise RuntimeError(
            f"Harmony encoder is not gpt-oss/{O200K_HARMONY}: stop_tokens={stops}. "
            "cl100k_base fallback is forbidden."
        )
    return enc


def stop_ids_for_tool_actions(enc=None) -> list[int]:
    """IDs that end an assistant tool action. Always <|call|> and <|return|> only."""
    del enc
    return list(CANONICAL_STOP_TOKEN_IDS)


def decode_ids(enc, ids: Sequence[int]) -> str:
    try:
        text = enc.decode_utf8(list(ids))
    except Exception:
        try:
            text = enc.decode(list(ids))
        except Exception:
            text = ""
    # Offline/fallback tokenizers can yield isolated UTF-16 surrogates for
    # unknown byte sequences; traces must remain valid UTF-8 JSON.
    return str(text).encode("utf-8", "replace").decode("utf-8")


@dataclass
class ParsedToolCall:
    parsed: bool
    legal: bool
    tool_name: str | None
    arguments: dict[str, Any] | None
    parse_method: str
    raw_json: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loads_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # truncated / extra trailing tokens: take first balanced object
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def parse_harmony_tool_call(
    text: str,
    *,
    completion_ids: Sequence[int] | None = None,
    enc=None,
) -> ParsedToolCall:
    """Strict parser: canonical Harmony recipient + JSON args.

    Does NOT fall back to 'tool name mentioned in analysis prose'.
    """
    text = text or ""
    if completion_ids is not None:
        try:
            enc = enc or load_harmony_enc()
            from openai_harmony import Role

            msgs = enc.parse_messages_from_completion_tokens(
                list(completion_ids), Role.ASSISTANT
            )
            for msg in msgs or []:
                recipient = str(getattr(msg, "recipient", "") or "")
                channel = str(getattr(msg, "channel", "") or "")
                content = ""
                try:
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        parts = []
                        for c in content:
                            parts.append(getattr(c, "text", None) or str(c))
                        content = "".join(parts)
                    elif content is None:
                        content = str(msg)
                    else:
                        content = str(content)
                except Exception:
                    content = str(msg)
                name = None
                rec_l = recipient.lower()
                if rec_l.startswith("functions."):
                    name = recipient.split(".", 1)[-1]
                elif rec_l in SCHEMA_TOOLS or recipient in SCHEMA_TOOLS:
                    name = recipient
                else:
                    m = _TO_RE.search(recipient)
                    if m:
                        name = m.group("name")
                name = _canonicalize_tool_name(name)
                if name in ("functions", "None", "none"):
                    name = None
                if name and channel in ("commentary", "analysis", "") and (
                    rec_l.startswith("functions.") or name in SCHEMA_TOOLS
                ):
                    args = _loads_json(content)
                    legal = name in SCHEMA_TOOLS
                    return ParsedToolCall(
                        parsed=True,
                        legal=legal,
                        tool_name=name,
                        arguments=args,
                        parse_method="harmony_parse_messages",
                        raw_json=content[:2000] if content else None,
                        error=None if args is not None else "json_missing_or_invalid",
                    )
        except Exception as exc:  # noqa: BLE001
            harmony_err = str(exc)[:200]
        else:
            harmony_err = None
    else:
        harmony_err = None

    m = _TO_RE.search(text)
    name = _canonicalize_tool_name(m.group("name") if m else None)
    if name in ("functions", "None", "none"):
        name = None
    body = None
    mj = _MESSAGE_JSON_RE.search(text)
    if mj:
        body = mj.group("body")
    elif m:
        mj2 = _JSON_AFTER_TO_RE.search(text[m.start() :])
        if mj2:
            body = mj2.group("body")
    args = _loads_json(body) if body else None
    if name:
        legal = name in SCHEMA_TOOLS
        return ParsedToolCall(
            parsed=True,
            legal=legal,
            tool_name=name,
            arguments=args,
            parse_method="regex_to_functions",
            raw_json=(body or "")[:2000] if body else None,
            error=harmony_err or (None if args is not None else "json_missing_or_invalid"),
        )
    return ParsedToolCall(
        parsed=False,
        legal=False,
        tool_name=None,
        arguments=None,
        parse_method="none",
        raw_json=None,
        error=harmony_err or "no_harmony_recipient",
    )


def format_aware_char_mask(text: str) -> list[bool]:
    """Mask from the assistant tool-action Harmony control tokens through <|call|>."""
    n = len(text or "")
    mask = [False] * n
    if n == 0:
        return mask
    idx = text.find(" to=functions.")
    if idx < 0:
        idx = text.find("to=functions.")
    if idx < 0:
        idx = text.find("<|channel|>commentary")
    if idx < 0:
        return mask
    start_tag = text.rfind("<|start|>assistant", 0, idx + 1)
    start = start_tag if start_tag >= 0 else idx
    end = text.find("<|call|>", start)
    end = (end + len("<|call|>")) if end >= 0 else n
    for i in range(start, min(end, n)):
        mask[i] = True
    return mask


def recent_actions_obs(actions_obs: list[tuple[Any, Any]], *, keep: int = 12) -> list[tuple[Any, Any]]:
    """Keep the latest tool-call / observation pairs so prompts stay in context."""
    if keep <= 0 or len(actions_obs) <= keep:
        return list(actions_obs)
    return list(actions_obs[-int(keep) :])


def fit_prompt_ids_to_context(
    ids: list[int] | tuple[int, ...],
    *,
    max_model_len: int,
    max_new_tokens: int = 1,
    keep_prefix: int = 4096,
) -> list[int]:
    """Drop the middle of an overlong Harmony prompt; keep system prefix + recent tail."""
    tokens = [int(x) for x in ids]
    budget = max(1, int(max_model_len) - max(1, int(max_new_tokens)))
    if len(tokens) <= budget:
        return tokens
    prefix = min(max(0, int(keep_prefix)), budget // 3)
    tail = budget - prefix
    if prefix <= 0 or tail <= 0:
        return tokens[-budget:]
    return tokens[:prefix] + tokens[-tail:]


def build_first_turn_prompt_ids(query: str, enc=None) -> list[int]:
    _ensure_scope()
    from openai_harmony import Role
    from harness.ultra_core import build_context, get_system_prompt

    enc = enc or load_harmony_enc()
    conv = build_context(get_system_prompt(query), None, [], [])
    return [int(x) for x in enc.render_conversation_for_completion(conv, Role.ASSISTANT)]


def build_continuation_prompt_ids(
    query: str,
    *,
    actions_obs: list[tuple[Any, Any]],
    wm_text: str | None = None,
    enc=None,
) -> list[int]:
    _ensure_scope()
    from openai_harmony import Role
    from harness.ultra_core import build_context, get_system_prompt

    enc = enc or load_harmony_enc()
    actions_obs = recent_actions_obs(actions_obs, keep=12)
    actions = [a for a, _ in actions_obs]
    obs = [o for _, o in actions_obs]
    conv = build_context(get_system_prompt(query), wm_text, actions, obs)
    return [int(x) for x in enc.render_conversation_for_completion(conv, Role.ASSISTANT)]


def make_action(tool_name: str, params: dict[str, Any], reasoning: str | None = None):
    _ensure_scope()
    from harness.tools import SerializedTool, ToolSchema
    from harness.trajectory import Action

    schema = ToolSchema(
        name=tool_name,
        description=tool_name,
        parameters={"type": "object"},
        required=[],
    )
    return Action(
        tools=[SerializedTool(tool_schema=schema)],
        params=[params],
        sources=["call_1"],
        reasoning=reasoning,
    )


def make_observation(text: str):
    _ensure_scope()
    from harness.trajectory import Observation

    return Observation(observations=[text], sources=["call_1"], tool_metadata=[None])


def canonical_examples() -> dict[str, str]:
    valid = (
        "<|channel|>analysis<|message|>Need a corpus search.<|end|>"
        "<|start|>assistant to=functions.search_corpus<|channel|>commentary "
        "<|constrain|>json<|message|>{\"query\": \"Apple FY2023 10-K filing date\"}<|call|>"
    )
    end_search = (
        "<|channel|>analysis<|message|>Enough evidence.<|end|>"
        "<|start|>assistant to=functions.end_search<|channel|>commentary "
        "<|constrain|>json<|message|>{\"reasoning\": \"curated set is sufficient\"}<|call|>"
    )
    malformed = (
        "<|channel|>analysis<|message|>Bad tool.<|end|>"
        "<|start|>assistant to=functions.not_a_real_tool<|channel|>commentary "
        "<|constrain|>json<|message|>{\"query\": \"x\"}<|call|>"
    )
    prose_only = (
        "<|channel|>analysis<|message|>I should call search_corpus next but I will not.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>search_corpus is useful.<|return|>"
    )
    return {
        "canonical_valid_search": valid,
        "canonical_end_search": end_search,
        "malformed_tool_name": malformed,
        "analysis_prose_only": prose_only,
    }


def run_parser_contract_tests() -> dict[str, Any]:
    examples = canonical_examples()
    rows = []
    p_valid = parse_harmony_tool_call(examples["canonical_valid_search"])
    rows.append(
        {
            "id": "canonical_valid_harmony_tool_call",
            "expect": "pass",
            "parsed": p_valid.parsed,
            "legal": p_valid.legal,
            "tool_name": p_valid.tool_name,
            "ok": bool(p_valid.parsed and p_valid.legal and p_valid.tool_name == "search_corpus"),
        }
    )
    p_end = parse_harmony_tool_call(examples["canonical_end_search"])
    rows.append(
        {
            "id": "canonical_end_search_call",
            "expect": "pass",
            "parsed": p_end.parsed,
            "legal": p_end.legal,
            "tool_name": p_end.tool_name,
            "ok": bool(p_end.parsed and p_end.legal and p_end.tool_name == "end_search"),
        }
    )
    p_bad = parse_harmony_tool_call(examples["malformed_tool_name"])
    rows.append(
        {
            "id": "malformed_tool_name",
            "expect": "fail_legal",
            "parsed": p_bad.parsed,
            "legal": p_bad.legal,
            "tool_name": p_bad.tool_name,
            "ok": bool(p_bad.parsed and (not p_bad.legal) and p_bad.tool_name == "not_a_real_tool"),
        }
    )
    p_prose = parse_harmony_tool_call(examples["analysis_prose_only"])
    rows.append(
        {
            "id": "analysis_prose_must_not_count_as_tool",
            "expect": "unparsed",
            "parsed": p_prose.parsed,
            "legal": p_prose.legal,
            "tool_name": p_prose.tool_name,
            "ok": (not p_prose.parsed) and (p_prose.tool_name is None),
        }
    )
    return {
        "n": len(rows),
        "n_pass": sum(1 for r in rows if r["ok"]),
        "all_ok": all(r["ok"] for r in rows),
        "rows": rows,
    }


def generate_tool_turn(
    model,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int = 384,
    enc=None,
    device=None,
) -> dict[str, Any]:
    import torch

    enc = enc or load_harmony_enc()
    stop_ids = stop_ids_for_tool_actions(enc)
    if device is None:
        device = next(model.parameters()).device
    inp = torch.tensor([list(prompt_ids)], device=device)
    attn = torch.ones_like(inp)
    with torch.no_grad():
        gen = model.generate(
            inp,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=stop_ids[0],
        )
    new_ids = gen[0, inp.size(1) :].tolist()
    text = decode_ids(enc, new_ids)
    parsed = parse_harmony_tool_call(text, completion_ids=new_ids, enc=enc)
    term = "max_new_tokens"
    if new_ids:
        last = int(new_ids[-1])
        if last == 200012:
            term = "call"
        elif last == 200002:
            term = "return"
        elif last == 200007:
            term = "end"
    return {
        "completion_ids": new_ids,
        "text": text,
        "n_tokens": len(new_ids),
        "termination": term,
        "parsed": parsed.to_dict(),
    }
