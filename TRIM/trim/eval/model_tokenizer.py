"""Model-family tokenizer / prompt runtime for TRIM eval.

gpt-oss uses Harmony (o200k) token IDs and ``to=functions.*`` tool calls.
Qwen3 uses the HF chat template, ``<|im_end|>`` stops, and ``<tool_call>`` XML.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

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

FAMILY_GPTOSS = "gpt-oss"
FAMILY_QWEN3 = "qwen3"
QWEN3_CHAT = "qwen3_chat"
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
    name = str(source or "").lower()
    if "qwen" in name:
        return FAMILY_QWEN3
    if tokenizer is not None:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        call = tokenizer.convert_tokens_to_ids("<|call|>")
        vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
        try:
            vocab = max(vocab, int(len(tokenizer) or 0))
        except (TypeError, AttributeError):
            pass
        if im_end in {QWEN3_IM_END_ID} and call not in {200012}:
            return FAMILY_QWEN3
        if vocab and vocab < 180000 and im_end not in {None, -1}:
            return FAMILY_QWEN3
        if call == 200012:
            return FAMILY_GPTOSS
    return FAMILY_GPTOSS


def encoding_config_for_model(model_path: str) -> dict[str, Any]:
    family = detect_model_family(model_path)
    if family == FAMILY_QWEN3:
        return {
            "family": FAMILY_QWEN3,
            "encoding": QWEN3_CHAT,
            "stop_token_ids": [QWEN3_IM_END_ID],
        }
    return {
        "family": FAMILY_GPTOSS,
        "encoding": O200K_HARMONY,
        "stop_token_ids": list(CANONICAL_STOP_TOKEN_IDS),
    }


def assert_qwen3_prompt_ids(ids: Sequence[int], *, what: str = "prompt") -> list[int]:
    """Refuse Harmony / ASCII-fallback IDs on the Qwen3 chat path."""
    tokens = [int(x) for x in ids]
    if not tokens:
        raise RuntimeError(f"{what} is empty; Qwen3 chat prompt IDs are required")
    if prompt_ids_are_character_fallback(tokens):
        raise RuntimeError(
            f"{what} looks like the local Harmony character fallback "
            f"(first20={tokens[:20]}). Qwen3 eval must use the chat-template tokenizer, "
            f"not ord('[Role.SYSTEM]...')."
        )
    has_im_start = QWEN3_IM_START_ID in tokens
    has_im_end = QWEN3_IM_END_ID in tokens
    looks_harmony = tokens[0] == HARMONY_START_ID or (
        HARMONY_START_ID in tokens[:8] and not has_im_start
    )
    if looks_harmony:
        raise RuntimeError(
            f"{what} looks like gpt-oss Harmony IDs sent to Qwen3 "
            f"(first20={tokens[:20]}). Expected <|im_start|>={QWEN3_IM_START_ID}."
        )
    if not has_im_start and not has_im_end:
        raise RuntimeError(
            f"{what} is not a Qwen3 chat prompt: first20={tokens[:20]}. "
            f"Expected <|im_start|>={QWEN3_IM_START_ID}."
        )
    return tokens


def assert_family_prompt_ids(
    ids: Sequence[int],
    *,
    family: str,
    what: str = "prompt",
) -> list[int]:
    name = str(family or "").lower()
    if name in {FAMILY_QWEN3, QWEN3_CHAT, "qwen"}:
        return assert_qwen3_prompt_ids(ids, what=what)
    return assert_o200k_harmony_token_ids(ids, what=what)


def assert_qwen3_tokenizer(tokenizer: Any, *, source: str) -> dict[str, Any]:
    name = str(
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "name", None)
        or source
        or ""
    ).lower()
    for marker in FORBIDDEN_TOKENIZER_MARKERS:
        if marker in name:
            raise RuntimeError(
                f"tokenizer {name!r} looks like {marker}; Qwen3 chat tokenizer is required"
            )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_end in {None, -1} or im_start in {None, -1}:
        raise RuntimeError(
            f"tokenizer {source!r} is missing Qwen <|im_start|>/<|im_end|> specials"
        )
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    try:
        tokenizer_len = int(len(tokenizer) or 0)
    except (TypeError, AttributeError):
        tokenizer_len = 0
    effective = max(vocab_size, tokenizer_len)
    if effective < 100000 or effective > 200000:
        raise RuntimeError(
            f"tokenizer effective_vocab_size={effective} does not look like Qwen3. source={source}"
        )
    return {
        "encoding": QWEN3_CHAT,
        "family": FAMILY_QWEN3,
        "source": source,
        "vocab_size": vocab_size,
        "effective_vocab_size": effective,
        "special_token_ids": {"<|im_end|>": int(im_end), "<|im_start|>": int(im_start)},
        "stop_token_ids": [int(im_end)],
    }


def assert_model_tokenizer(tokenizer: Any, *, source: str) -> dict[str, Any]:
    from trim.training.vllm_hybrid import assert_gptoss_tokenizer

    family = detect_model_family(source, tokenizer)
    if family == FAMILY_QWEN3:
        return assert_qwen3_tokenizer(tokenizer, source=source)
    return assert_gptoss_tokenizer(tokenizer, source=source)


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
    match = _TOOL_CALL_RE.search(text)
    raw = match.group(1) if match else None
    if raw is None:
        # Some Qwen dumps emit a bare JSON object with name/arguments.
        obj = _loads_json(text)
        if isinstance(obj, dict) and (obj.get("name") or obj.get("function")):
            raw_obj = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        else:
            return ParsedToolCall(
                parsed=False,
                legal=False,
                tool_name=None,
                arguments=None,
                parse_method="none",
                raw_json=None,
                error="no_qwen_tool_call",
            )
    else:
        raw_obj = _loads_json(raw)
    if not isinstance(raw_obj, dict):
        return ParsedToolCall(
            parsed=True,
            legal=False,
            tool_name=None,
            arguments=None,
            parse_method="qwen_tool_call",
            raw_json=(raw or text)[:2000],
            error="json_missing_or_invalid",
        )
    name = _canonicalize_tool_name(raw_obj.get("name") or raw_obj.get("tool_name"))
    args = raw_obj.get("arguments") or raw_obj.get("parameters") or {}
    if isinstance(args, str):
        args = _loads_json(args) or {}
    if not isinstance(args, dict):
        args = {}
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
        if self.family == FAMILY_QWEN3 and self.tokenizer is not None:
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            return str(text).encode("utf-8", "replace").decode("utf-8")
        enc = self.harmony
        if enc is None:
            return ""
        return decode_ids(enc, tokens)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        if self.family == FAMILY_QWEN3:
            if self.tokenizer is None:
                raise RuntimeError("Qwen3 encoding is missing a Hugging Face tokenizer")
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
        if self.family == FAMILY_QWEN3:
            return self._qwen_prompt_ids(query, [])
        from trim.eval.harmony_runtime import build_first_turn_prompt_ids

        return build_first_turn_prompt_ids(query, enc=self.harmony)

    def build_continuation_prompt_ids(
        self,
        query: str,
        *,
        actions_obs: list[tuple[Any, Any]],
        wm_text: str | None = None,
    ) -> list[int]:
        if self.family == FAMILY_QWEN3:
            return self._qwen_prompt_ids(query, actions_obs, wm_text=wm_text)
        from trim.eval.harmony_runtime import build_continuation_prompt_ids

        return build_continuation_prompt_ids(
            query, actions_obs=actions_obs, wm_text=wm_text, enc=self.harmony
        )

    def parse_tool_call(self, text: str, completion_ids: Sequence[int] | None = None):
        if self.family == FAMILY_QWEN3:
            return parse_qwen_tool_call(text)
        return parse_harmony_tool_call(
            text, completion_ids=completion_ids, enc=self.harmony
        )

    def _qwen_prompt_ids(
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
            raise RuntimeError("Qwen3 encoding is missing a Hugging Face tokenizer")
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
        return assert_qwen3_prompt_ids(_to_token_ids(raw), what="Qwen3 chat prompt")


def load_model_encoding(model_path: str | None = None) -> ModelEncoding:
    source = str(model_path or "")
    family = detect_model_family(source)
    if family == FAMILY_QWEN3:
        if not source:
            raise RuntimeError("Qwen3 eval requires --model_name / a local checkpoint path")
        patch_transformers_tokenizer_compat()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        audit = assert_qwen3_tokenizer(tokenizer, source=source)
        return ModelEncoding(
            family=FAMILY_QWEN3,
            source=source,
            encoding_name=QWEN3_CHAT,
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
