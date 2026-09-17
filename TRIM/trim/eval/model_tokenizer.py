"""Model-family tokenizer / prompt runtime for TRIM eval.

gpt-oss uses Harmony (o200k) token IDs and ``to=functions.*`` tool calls.
Qwen3 / Qwen3.5 / GLM-4 / Gemma-3 and other HF instruct models use the chat
template, model-specific stop tokens, and ``<tool_call>``-style tool output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from trim.eval.model_profiles import (
    FAMILY_GPTOSS,
    FAMILY_HF_CHAT,
    FAMILY_QWEN3,
    STACK_HARMONY,
    is_hf_chat_family,
    resolve_model_profile,
)
from trim.eval.harmony_runtime import (
    CANONICAL_STOP_TOKEN_IDS,
    HARMONY_START_ID,
    O200K_HARMONY,
    SCHEMA_TOOLS,
    _canonicalize_tool_name,
    _loads_json,
    assert_o200k_harmony_token_ids,
    decode_ids,
    load_harmony_enc,
    parse_harmony_tool_call,
    prompt_ids_are_character_fallback,
)

FORBIDDEN_TOKENIZER_MARKERS = ("cl100k", "r50k", "p50k", "gpt2")

QWEN3_CHAT = "qwen3_chat"
HF_CHAT_TOOLS = "hf_chat_tools"
QWEN3_IM_END_ID = 151645
QWEN3_IM_START_ID = 151644

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def patch_transformers_tokenizer_compat() -> None:
    """vLLM 0.19 still reads ``all_special_tokens_extended``; Transformers 5 dropped it.

    Applies to every TokenizersBackend subclass, including Qwen2Tokenizer / Qwen3.
    """
    try:
        from transformers import TokenizersBackend

        if not hasattr(TokenizersBackend, "all_special_tokens_extended"):
            TokenizersBackend.all_special_tokens_extended = property(
                lambda self: list(self.all_special_tokens)
            )
    except ImportError:
        pass


def detect_model_family(source: str, tokenizer: Any | None = None) -> str:
    return resolve_model_profile(source, tokenizer).family


def encoding_config_for_model(model_path: str) -> dict[str, Any]:
    profile = resolve_model_profile(model_path)
    if profile.stack == STACK_HARMONY:
        return {
            "family": FAMILY_GPTOSS,
            "encoding": O200K_HARMONY,
            "stop_token_ids": list(CANONICAL_STOP_TOKEN_IDS),
        }
    if profile.family == FAMILY_QWEN3:
        return {
            "family": FAMILY_QWEN3,
            "encoding": QWEN3_CHAT,
            "stop_token_ids": [QWEN3_IM_END_ID],
        }
    return {
        "family": FAMILY_HF_CHAT,
        "encoding": HF_CHAT_TOOLS,
        "stop_token_ids": [],
    }


def _reject_harmony_or_ascii_fallback(ids: Sequence[int], *, what: str) -> list[int]:
    tokens = [int(x) for x in ids]
    if not tokens:
        raise RuntimeError(f"{what} is empty; HF chat prompt IDs are required")
    if prompt_ids_are_character_fallback(tokens):
        raise RuntimeError(
            f"{what} looks like the local Harmony character fallback "
            f"(first20={tokens[:20]}). HF chat eval must use the chat-template tokenizer, "
            f"not ord('[Role.SYSTEM]...')."
        )
    has_im_start = QWEN3_IM_START_ID in tokens
    looks_harmony = tokens[0] == HARMONY_START_ID or (
        HARMONY_START_ID in tokens[:8] and not has_im_start
    )
    if looks_harmony:
        raise RuntimeError(
            f"{what} looks like gpt-oss Harmony IDs sent to an HF chat model "
            f"(first20={tokens[:20]})."
        )
    return tokens


def assert_hf_chat_prompt_ids(
    ids: Sequence[int],
    *,
    what: str = "prompt",
    strict_qwen: bool = False,
) -> list[int]:
    """Refuse Harmony / ASCII-fallback IDs on HF chat-template paths."""
    tokens = _reject_harmony_or_ascii_fallback(ids, what=what)
    if not strict_qwen:
        return tokens
    has_im_start = QWEN3_IM_START_ID in tokens
    has_im_end = QWEN3_IM_END_ID in tokens
    if not has_im_start and not has_im_end:
        raise RuntimeError(
            f"{what} is not a Qwen3 chat prompt: first20={tokens[:20]}. "
            f"Expected <|im_start|>={QWEN3_IM_START_ID}."
        )
    return tokens


def assert_qwen3_prompt_ids(ids: Sequence[int], *, what: str = "prompt") -> list[int]:
    """Backward-compatible alias: strict Qwen3 chat prompt validation."""
    return assert_hf_chat_prompt_ids(ids, what=what, strict_qwen=True)


def assert_family_prompt_ids(
    ids: Sequence[int],
    *,
    family: str,
    what: str = "prompt",
    source: str = "",
) -> list[int]:
    name = str(family or "").lower()
    if is_hf_chat_family(name):
        profile = resolve_model_profile(source or name)
        return assert_hf_chat_prompt_ids(
            ids,
            what=what,
            strict_qwen=profile.strict_qwen_prompt_ids,
        )
    return assert_o200k_harmony_token_ids(ids, what=what)


def _effective_vocab_size(tokenizer: Any) -> int:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    try:
        tokenizer_len = int(len(tokenizer) or 0)
    except (TypeError, AttributeError):
        tokenizer_len = 0
    return max(vocab_size, tokenizer_len)


def _resolve_stop_token_ids(tokenizer: Any) -> list[int]:
    for attr in ("eos_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid not in {None, -1}:
            return [int(tid)]
    for tok in ("<|im_end|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid not in {None, -1}:
            return [int(tid)]
    return []


def assert_qwen3_tokenizer(tokenizer: Any, *, source: str) -> dict[str, Any]:
    return assert_hf_chat_tokenizer(tokenizer, source=source, strict_qwen=True)


def assert_hf_chat_tokenizer(
    tokenizer: Any,
    *,
    source: str,
    strict_qwen: bool = False,
) -> dict[str, Any]:
    profile = resolve_model_profile(source, tokenizer)
    name = str(
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "name", None)
        or source
        or ""
    ).lower()
    for marker in FORBIDDEN_TOKENIZER_MARKERS:
        if marker in name:
            raise RuntimeError(
                f"tokenizer {name!r} looks like {marker}; HF chat tokenizer is required"
            )
    try:
        call = tokenizer.convert_tokens_to_ids("<|call|>")
    except Exception:
        call = None
    if call == 200012:
        raise RuntimeError(
            f"tokenizer {source!r} looks like gpt-oss Harmony (<|call|>=200012); "
            "HF chat tokenizer is required"
        )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    effective = _effective_vocab_size(tokenizer)
    if strict_qwen or profile.strict_qwen_prompt_ids:
        if im_end in {None, -1} or im_start in {None, -1}:
            raise RuntimeError(
                f"tokenizer {source!r} is missing Qwen <|im_start|>/<|im_end|> specials"
            )
        if effective < 100000 or effective > 200000:
            raise RuntimeError(
                f"tokenizer effective_vocab_size={effective} does not look like Qwen3. source={source}"
            )
        stop_token_ids = [int(im_end)]
        special_token_ids = {
            "<|im_end|>": int(im_end),
            "<|im_start|>": int(im_start),
        }
        encoding = QWEN3_CHAT
        family = FAMILY_QWEN3
    else:
        if effective < 32000:
            raise RuntimeError(
                f"tokenizer effective_vocab_size={effective} is too small for HF chat. source={source}"
            )
        stop_token_ids = _resolve_stop_token_ids(tokenizer)
        if not stop_token_ids and im_end not in {None, -1}:
            stop_token_ids = [int(im_end)]
        special_token_ids = {}
        if im_end not in {None, -1}:
            special_token_ids["<|im_end|>"] = int(im_end)
        if im_start not in {None, -1}:
            special_token_ids["<|im_start|>"] = int(im_start)
        encoding = HF_CHAT_TOOLS
        family = FAMILY_HF_CHAT
    return {
        "encoding": encoding,
        "family": family,
        "source": source,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "effective_vocab_size": effective,
        "special_token_ids": special_token_ids,
        "stop_token_ids": stop_token_ids,
        "profile_label": profile.label,
    }


def assert_model_tokenizer(tokenizer: Any, *, source: str) -> dict[str, Any]:
    from trim.training.vllm_hybrid import assert_gptoss_tokenizer

    profile = resolve_model_profile(source, tokenizer)
    if profile.stack == STACK_HARMONY:
        return assert_gptoss_tokenizer(tokenizer, source=source)
    return assert_hf_chat_tokenizer(
        tokenizer,
        source=source,
        strict_qwen=profile.strict_qwen_prompt_ids,
    )


def qwen_tool_schemas() -> list[dict[str, Any]]:
    def fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    return [
        fn(
            "fan_out_search",
            "Run up to 8 diverse queries in parallel.",
            {"queries": {"type": "array", "items": {"type": "string"}}},
            ["queries"],
        ),
        fn(
            "search_corpus",
            "Single semantic + keyword search.",
            {"query": {"type": "string"}},
            ["query"],
        ),
        fn(
            "grep_corpus",
            "Exact regex pattern matching on the corpus.",
            {"pattern": {"type": "string"}},
            ["pattern"],
        ),
        fn(
            "read_document",
            "Read a document's full content.",
            {"doc_id": {"type": "string"}},
            ["doc_id"],
        ),
        fn(
            "review_docs",
            "Re-read previously found documents from memory.",
            {"doc_ids": {"type": "array", "items": {"type": "string"}}},
            ["doc_ids"],
        ),
        fn(
            "curate",
            "Update the curated set of relevant documents.",
            {
                "add_ids": {"type": "array", "items": {"type": "string"}},
                "remove_ids": {"type": "array", "items": {"type": "string"}},
            },
            ["add_ids"],
        ),
        fn(
            "end_search",
            "Submit the curated set and conclude.",
            {"reasoning": {"type": "string"}},
            [],
        ),
    ]


def parse_qwen_tool_call(text: str, completion_ids: Sequence[int] | None = None, **_kwargs):
    from trim.eval.harmony_runtime import ParsedToolCall

    text = text or ""
    matches = list(_TOOL_CALL_RE.finditer(text))
    raw = None
    if len(matches) > 1:
        return ParsedToolCall(
            parsed=False,
            legal=False,
            tool_name=None,
            arguments=None,
            parse_method="qwen_tool_call",
            raw_json=text[:2000] if text else None,
            error="multiple_qwen_tool_calls",
        )
    if len(matches) == 1:
        raw = matches[0].group(1)
    else:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            raw = stripped
        else:
            return ParsedToolCall(
                parsed=False,
                legal=False,
                tool_name=None,
                arguments=None,
                parse_method="qwen_tool_call",
                raw_json=text[:2000] if text else None,
                error="no_qwen_tool_call",
            )
    raw_obj = _loads_json(raw)
    if not isinstance(raw_obj, dict):
        return ParsedToolCall(
            parsed=False,
            legal=False,
            tool_name=None,
            arguments=None,
            parse_method="qwen_tool_call",
            raw_json=(raw or text)[:2000],
            error="json_missing_or_invalid",
        )
    name = _canonicalize_tool_name(
        raw_obj.get("name") or raw_obj.get("tool_name") or raw_obj.get("tool")
    )
    args = raw_obj.get("arguments") or raw_obj.get("parameters") or {}
    if isinstance(args, str):
        loaded = _loads_json(args)
        if not isinstance(loaded, dict):
            return ParsedToolCall(
                parsed=False,
                legal=False,
                tool_name=name,
                arguments=None,
                parse_method="qwen_tool_call",
                raw_json=(raw or text)[:2000],
                error="json_missing_or_invalid",
            )
        args = loaded
    if not isinstance(args, dict):
        return ParsedToolCall(
            parsed=False,
            legal=False,
            tool_name=name,
            arguments=None,
            parse_method="qwen_tool_call",
            raw_json=(raw or text)[:2000],
            error="json_missing_or_invalid",
        )
    legal = bool(name) and name in SCHEMA_TOOLS
    return ParsedToolCall(
        parsed=True,
        legal=legal,
        tool_name=name,
        arguments=args,
        parse_method="qwen_tool_call",
        raw_json=(raw or json.dumps(raw_obj, ensure_ascii=False))[:2000],
        error=None if args is not None else "json_missing_or_invalid",
    )


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


def _to_token_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("input_ids")
    if hasattr(raw, "input_ids"):
        raw = raw.input_ids
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if raw and isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    if isinstance(raw, int):
        return [int(raw)]
    return [int(x) for x in list(raw)]


@dataclass
class ModelEncoding:
    family: str
    source: str
    encoding_name: str
    stop_token_ids: list[int]
    tokenizer: Any | None = None
    harmony: Any | None = None

    def decode_tokens(self, ids: Sequence[int]) -> str:
        tokens = [int(x) for x in ids]
        if is_hf_chat_family(self.family) and self.tokenizer is not None:
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            return str(text).encode("utf-8", "replace").decode("utf-8")
        enc = self.harmony
        if enc is None:
            return ""
        return decode_ids(enc, tokens)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        if is_hf_chat_family(self.family):
            if self.tokenizer is None:
                raise RuntimeError("HF chat encoding is missing a Hugging Face tokenizer")
            try:
                return _to_token_ids(
                    self.tokenizer.encode(str(text), add_special_tokens=False)
                )
            except TypeError:
                return _to_token_ids(self.tokenizer.encode(str(text)))
        enc = self.harmony
        if enc is None or not hasattr(enc, "encode"):
            raise RuntimeError("Harmony encoding cannot encode raw text")
        try:
            return [int(x) for x in enc.encode(str(text), allowed_special="all", **kwargs)]
        except TypeError:
            return [int(x) for x in enc.encode(str(text), **kwargs)]

    def build_first_turn_prompt_ids(self, query: str) -> list[int]:
        if is_hf_chat_family(self.family):
            return self._hf_chat_prompt_ids(query, [])
        from trim.eval.harmony_runtime import build_first_turn_prompt_ids

        return build_first_turn_prompt_ids(query, enc=self.harmony)

    def build_continuation_prompt_ids(
        self,
        query: str,
        *,
        actions_obs: list[tuple[Any, Any]],
        wm_text: str | None = None,
    ) -> list[int]:
        if is_hf_chat_family(self.family):
            return self._hf_chat_prompt_ids(query, actions_obs, wm_text=wm_text)
        from trim.eval.harmony_runtime import build_continuation_prompt_ids

        return build_continuation_prompt_ids(
            query, actions_obs=actions_obs, wm_text=wm_text, enc=self.harmony
        )

    def parse_tool_call(self, text: str, completion_ids: Sequence[int] | None = None):
        if is_hf_chat_family(self.family):
            return parse_qwen_tool_call(text)
        return parse_harmony_tool_call(
            text, completion_ids=completion_ids, enc=self.harmony
        )

    def _hf_chat_prompt_ids(
        self,
        query: str,
        actions_obs: list[tuple[Any, Any]],
        *,
        wm_text: str | None = None,
    ) -> list[int]:
        from trim.eval.harmony_runtime import _ensure_scope, recent_actions_obs

        _ensure_scope()
        from harness.ultra_core import get_system_prompt

        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("HF chat encoding is missing a Hugging Face tokenizer")
        profile = resolve_model_profile(self.source, tokenizer)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Follow the retrieval-subagent instructions and call tools."},
            {"role": "user", "content": get_system_prompt(query)},
        ]
        if wm_text:
            messages.append({"role": "user", "content": str(wm_text)})
        for action, obs in recent_actions_obs(list(actions_obs), keep=12):
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
                                "arguments": args,
                            },
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "content": _obs_text(obs)})
        kwargs: dict[str, Any] = {
            "tools": qwen_tool_schemas(),
            "add_generation_prompt": True,
            "tokenize": True,
        }
        try:
            raw = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            raw = tokenizer.apply_chat_template(messages, **kwargs)
        return assert_hf_chat_prompt_ids(
            _to_token_ids(raw),
            what=f"{profile.label} chat prompt",
            strict_qwen=profile.strict_qwen_prompt_ids,
        )


def load_model_encoding(model_path: str | None = None) -> ModelEncoding:
    source = str(model_path or "")
    profile = resolve_model_profile(source)
    if profile.stack != STACK_HARMONY:
        if not source:
            raise RuntimeError("HF chat eval requires --model_name / a local checkpoint path")
        patch_transformers_tokenizer_compat()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        audit = assert_hf_chat_tokenizer(
            tokenizer,
            source=source,
            strict_qwen=profile.strict_qwen_prompt_ids,
        )
        return ModelEncoding(
            family=str(audit["family"]),
            source=source,
            encoding_name=str(audit["encoding"]),
            stop_token_ids=list(audit["stop_token_ids"]),
            tokenizer=tokenizer,
        )
    return ModelEncoding(
        family=FAMILY_GPTOSS,
        source=source or "harmony",
        encoding_name=O200K_HARMONY,
        stop_token_ids=list(CANONICAL_STOP_TOKEN_IDS),
        harmony=load_harmony_enc(source or None),
    )
