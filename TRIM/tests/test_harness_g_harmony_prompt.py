from __future__ import annotations

import json

import pytest

from trim.eval.harmony_runtime import HARMONY_START_ID, assert_o200k_harmony_token_ids
from trim.eval.harness_g_runtime import (
    build_harness_g_conversation,
    build_prompt_ids,
    render_prompt,
)
from trim.eval.model_tokenizer import FAMILY_QWEN3, QWEN3_IM_START_ID, assert_qwen3_prompt_ids


class _FakeHarmony:
    def __init__(self) -> None:
        self.conversations: list[object] = []
        self.encode_calls = 0

    def render_conversation_for_completion(self, conv, role):
        self.conversations.append((conv, role))
        return [HARMONY_START_ID, 17360, 200008]

    def encode(self, text, **_kwargs):
        del text
        self.encode_calls += 1
        raise AssertionError("raw encode() must not be used for gpt-oss Harness-G")


class _GptOssEnc:
    family = "gpt-oss"
    tokenizer = None

    def __init__(self, harmony: _FakeHarmony) -> None:
        self.harmony = harmony

    def encode(self, text, **_kwargs):
        del text
        raise AssertionError("ModelEncoding.encode must not be used for Harness-G gpt-oss")


class _QwenTok:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        del messages, add_generation_prompt
        if tokenize:
            return [QWEN3_IM_START_ID, 100, 200, 151645]
        return "<|im_start|>user\n<|im_end|>"


class _QwenEnc:
    family = FAMILY_QWEN3
    tokenizer = _QwenTok()
    harmony = None


def _conversation_tool_names(conv) -> set[str]:
    blob = json.loads(conv.to_json()) if hasattr(conv, "to_json") else conv
    names: set[str] = set()
    for msg in blob.get("messages") or []:
        for item in msg.get("content") or []:
            if not isinstance(item, dict):
                continue
            tools = item.get("tools") or {}
            functions = tools.get("functions") or {}
            for tool in functions.get("tools") or []:
                if isinstance(tool, dict) and tool.get("name"):
                    names.add(str(tool["name"]))
    return names


def test_harness_g_conversation_uses_g_tools_not_harness1():
    conv = build_harness_g_conversation("When was Apple founded?", "[WM]\nstep=0")
    names = _conversation_tool_names(conv)
    assert names == {"init", "select", "lookup", "answer"}
    assert "search_corpus" not in names
    assert "answer_with" not in names


def test_harness_g_conversation_can_include_answer_with():
    conv = build_harness_g_conversation(
        "When was Apple founded?",
        "[WM]\nstep=0",
        include_answer_with=True,
    )
    assert "answer_with" in _conversation_tool_names(conv)


def test_gptoss_build_prompt_ids_uses_harmony_renderer_not_raw_encode():
    harmony = _FakeHarmony()
    ids = build_prompt_ids("When was Apple founded?", "[WM]\nstep=0", _GptOssEnc(harmony))
    assert ids[0] == HARMONY_START_ID
    assert harmony.encode_calls == 0
    assert len(harmony.conversations) == 1
    conv, _role = harmony.conversations[0]
    assert _conversation_tool_names(conv) == {"init", "select", "lookup", "answer"}


def test_gptoss_missing_harmony_backend_refuses_raw_encode():
    class _Bare:
        family = "gpt-oss"

        def encode(self, text):
            return [ord(c) for c in text[:32]]

    with pytest.raises(RuntimeError, match="raw-text encode"):
        build_prompt_ids("When was Apple founded?", "[WM]", _Bare())


def test_qwen3_build_prompt_ids_still_uses_chat_template():
    ids = build_prompt_ids("When was Apple founded?", "[WM]\nstep=0", _QwenEnc())
    assert ids[0] == QWEN3_IM_START_ID
    assert_qwen3_prompt_ids(ids, what="Harness-G Qwen3 prompt")


def test_raw_text_prompt_would_fail_harmony_gate():
    """The previous Harness-G path encoded English text; vLLM rejected it."""
    from pathlib import Path

    model = Path("/data/ppnm/models/harness-1")
    if not (model / "tokenizer.json").is_file():
        pytest.skip("gpt-oss tokenizer not on disk")
    from trim.eval.harmony_runtime import load_harmony_enc
    from trim.eval.model_tokenizer import FAMILY_GPTOSS, ModelEncoding

    harmony = load_harmony_enc(str(model))
    enc = ModelEncoding(
        family=FAMILY_GPTOSS,
        source=str(model),
        encoding_name="o200k_harmony",
        stop_token_ids=[200012, 200002],
        harmony=harmony,
    )
    raw_ids = enc.encode(render_prompt("When was Apple founded?", "[WM]\nstep=0"))
    with pytest.raises(RuntimeError, match="not a gpt-oss Harmony prompt"):
        assert_o200k_harmony_token_ids(raw_ids, what="legacy Harness-G text prompt")


def test_gptoss_checkpoint_harness_g_prompt_starts_with_harmony_start():
    from pathlib import Path

    model = Path("/data/ppnm/models/harness-1")
    if not (model / "tokenizer.json").is_file():
        pytest.skip("gpt-oss tokenizer not on disk")
    from trim.eval.harmony_hf_encoding import render_harmony_conversation_text
    from trim.eval.harmony_runtime import decode_ids, load_harmony_enc
    from trim.eval.model_tokenizer import FAMILY_GPTOSS, ModelEncoding

    harmony = load_harmony_enc(str(model))
    enc = ModelEncoding(
        family=FAMILY_GPTOSS,
        source=str(model),
        encoding_name="o200k_harmony",
        stop_token_ids=[200012, 200002],
        harmony=harmony,
    )
    ids = build_prompt_ids(
        "When was Apple founded?",
        "[Harness-G Working Memory]\nstep=0 initialized=False",
        enc,
    )
    assert ids[0] == HARMONY_START_ID
    assert_o200k_harmony_token_ids(ids, what="Harness-G Harmony prompt")
    text = decode_ids(harmony, ids)
    assert text.startswith("<|start|>")
    conv = build_harness_g_conversation(
        "When was Apple founded?",
        "[Harness-G Working Memory]\nstep=0 initialized=False",
    )
    rendered = render_harmony_conversation_text(conv, next_role="assistant")
    assert "type init" in rendered
    assert "type select" in rendered
    assert "search_corpus" not in rendered
    assert "You are a Harness-G search agent" in rendered
