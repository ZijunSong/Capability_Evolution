"""Regression tests for multi-model profile routing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from trim.eval.model_profiles import (
    FAMILY_GPTOSS,
    FAMILY_HF_CHAT,
    FAMILY_QWEN3,
    is_harmony_model,
    needs_tool_call_smoke_test,
    resolve_model_profile,
    tool_call_parser_for,
    vllm_extra_shell,
)
from trim.eval.model_tokenizer import assert_hf_chat_tokenizer, detect_model_family
from trim.upstream_harness1.env_bridge import is_harmony_chat_model


class _GlmTok:
    name_or_path = "THUDM/glm-4-9b-chat"
    vocab_size = 151329
    eos_token_id = 151329

    def __len__(self) -> int:
        return 151329

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        if tok == "<|call|>":
            return -1
        if tok in {"<|im_end|>", "<|im_end|>"}:
            return 151329
        return -1


def test_resolve_qwen3_uses_hermes():
    profile = resolve_model_profile("Qwen/Qwen3-4B-Instruct-2507")
    assert profile.family == FAMILY_QWEN3
    assert profile.tool_call_parser == "hermes"
    assert needs_tool_call_smoke_test("Qwen3-4B-Instruct-2507")


def test_resolve_qwen35_uses_qwen3_coder():
    profile = resolve_model_profile("Qwen/Qwen3.5-35B-A3B")
    assert profile.family == FAMILY_HF_CHAT
    assert profile.tool_call_parser == "qwen3_coder"
    assert profile.reasoning_parser == "qwen3"


def test_resolve_glm45_and_glm47():
    assert resolve_model_profile("THUDM/glm-4-9b-chat").tool_call_parser == "glm45"
    assert resolve_model_profile("THUDM/GLM-4.7").tool_call_parser == "glm47"


def test_resolve_gemma_uses_pythonic():
    profile = resolve_model_profile("google/gemma-3-4b-it")
    assert profile.family == FAMILY_HF_CHAT
    assert profile.tool_call_parser == "pythonic"


def test_harmony_models_skip_smoke_test():
    assert resolve_model_profile("openai/gpt-oss-20b").family == FAMILY_GPTOSS
    assert not needs_tool_call_smoke_test("gpt-oss-20b")
    assert is_harmony_model("harness-1")
    assert is_harmony_chat_model("gpt-oss-20b")
    assert not is_harmony_chat_model("THUDM/glm-4-9b-chat")


def test_vllm_extra_shell_glm_includes_parser():
    extra = vllm_extra_shell("THUDM/glm-4-9b-chat")
    assert "--tool-call-parser glm45" in extra
    assert "openai" not in extra.split("--tool-call-parser")[1].split()[0]


def test_vllm_extra_shell_harness_includes_moe():
    extra = vllm_extra_shell("harness-1")
    assert "--tool-call-parser openai" in extra
    assert "--moe-backend triton" in extra


def test_assert_hf_chat_tokenizer_accepts_glm():
    audit = assert_hf_chat_tokenizer(_GlmTok(), source="THUDM/glm-4-9b-chat")
    assert audit["family"] == FAMILY_HF_CHAT
    assert audit["encoding"] == "hf_chat_tools"
    assert audit["stop_token_ids"] == [151329]


def test_detect_glm_family_from_path():
    assert detect_model_family("/data/models/glm-4-9b-chat") == FAMILY_HF_CHAT


def test_cli_vllm_extra_matches_python():
    trim_root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output(
        [sys.executable, "-m", "trim.eval.model_profiles", "vllm-extra", "Qwen3-4B-Instruct-2507"],
        cwd=str(trim_root),
        text=True,
    ).strip()
    assert out == vllm_extra_shell("Qwen3-4B-Instruct-2507")
    assert tool_call_parser_for("Qwen3-4B-Instruct-2507") == "hermes"
