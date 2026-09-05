from __future__ import annotations

from trim.eval.model_tokenizer import (
    FAMILY_GPTOSS,
    FAMILY_QWEN3,
    QWEN3_IM_START_ID,
    assert_family_prompt_ids,
    assert_model_tokenizer,
    assert_qwen3_prompt_ids,
    detect_model_family,
    encoding_config_for_model,
    parse_qwen_tool_call,
)
from trim.training.four_cell_runtime import parse_generated_action
from trim.training.vllm_hybrid import assert_gptoss_tokenizer


class _QwenTok:
    name_or_path = "/data/ppnm/models/Qwen3-4B-Instruct-2507"
    vocab_size = 151643

    def __len__(self) -> int:
        return 151669

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        return {"<|im_end|>": 151645, "<|im_start|>": 151644}.get(tok)


def test_detect_qwen3_from_path():
    assert detect_model_family("/data/ppnm/models/Qwen3-4B-Instruct-2507") == FAMILY_QWEN3
    assert detect_model_family("Qwen/Qwen3-4B-Instruct-2507") == FAMILY_QWEN3
    assert encoding_config_for_model("/data/ppnm/models/Qwen3-4B-Instruct-2507")["encoding"] == "qwen3_chat"
    assert encoding_config_for_model("/data/ppnm/models/harness-1")["encoding"] == "o200k_harmony"


def test_assert_model_tokenizer_accepts_qwen3():
    audit = assert_model_tokenizer(_QwenTok(), source="/data/ppnm/models/Qwen3-4B-Instruct-2507")
    assert audit["family"] == FAMILY_QWEN3
    assert audit["stop_token_ids"] == [151645]


def test_assert_gptoss_still_rejects_qwen_vocab():
    try:
        assert_gptoss_tokenizer(_QwenTok(), source="/data/ppnm/models/Qwen3-4B-Instruct-2507")
        raise AssertionError("expected gpt-oss assertion to reject Qwen")
    except RuntimeError as exc:
        assert "gpt-oss" in str(exc) or "Harmony" in str(exc) or "cl100k" in str(exc)


def test_parse_qwen_tool_call_xml():
    text = (
        "I will search now.\n"
        '<tool_call>\n{"name": "search_corpus", "arguments": {"query": "Apple 10-K"}}\n</tool_call>'
        "<|im_end|>"
    )
    parsed = parse_qwen_tool_call(text)
    assert parsed.parsed is True
    assert parsed.legal is True
    assert parsed.tool_name == "search_corpus"
    assert parsed.arguments == {"query": "Apple 10-K"}
    action, ok = parse_generated_action(text, None, enc=type("E", (), {"parse_tool_call": staticmethod(parse_qwen_tool_call)})())
    assert ok is True
    assert action["name"] == "search_corpus"


def test_parse_qwen_curate_channel_leak_name():
    text = '<tool_call>\n{"name": "curate?commentary", "arguments": {"add_ids": ["12"]}}\n</tool_call>'
    parsed = parse_qwen_tool_call(text)
    assert parsed.tool_name == "curate"
    assert parsed.legal is True
    assert parsed.arguments["add_ids"] == ["12"]


def test_qwen_prompt_ids_reject_harmony_and_ascii():
    import pytest
    from trim.eval.harmony_runtime import HARMONY_START_ID

    qwen_ids = [QWEN3_IM_START_ID, 100, 200, 151645, QWEN3_IM_START_ID]
    assert assert_qwen3_prompt_ids(qwen_ids)[0] == QWEN3_IM_START_ID
    assert assert_family_prompt_ids(qwen_ids, family=FAMILY_QWEN3)[0] == QWEN3_IM_START_ID
    with pytest.raises(RuntimeError, match="Harmony"):
        assert_qwen3_prompt_ids([HARMONY_START_ID, 17360, 200008, 200007])
    with pytest.raises(RuntimeError, match="character fallback"):
        assert_qwen3_prompt_ids(
            [91, 82, 111, 108, 101, 46, 83, 89, 83, 84, 69, 77, 93] + list(range(40, 100))
        )
    with pytest.raises(RuntimeError, match="not a gpt-oss Harmony prompt"):
        assert_family_prompt_ids(qwen_ids, family=FAMILY_GPTOSS)


def test_gptoss_path_is_not_classified_as_qwen():
    assert detect_model_family("openai/gpt-oss-20b") == FAMILY_GPTOSS
    assert detect_model_family("/mnt/songzijun/models/openai/gpt-oss-20b") == FAMILY_GPTOSS
    assert encoding_config_for_model("openai/gpt-oss-20b")["encoding"] == "o200k_harmony"


def test_one_episode_uses_model_encoding_prompt_builder():
    import inspect
    from trim.training import four_cell_runtime

    src = inspect.getsource(four_cell_runtime.one_episode)
    assert 'hasattr(enc, "build_first_turn_prompt_ids")' in src
    assert "enc.build_first_turn_prompt_ids(query)" in src
    assert "enc.build_continuation_prompt_ids" in src


def test_qwen_checkpoint_chat_prompt_if_present():
    from pathlib import Path

    model = Path("/data/ppnm/models/Qwen3-4B-Instruct-2507")
    if not (model / "tokenizer_config.json").is_file():
        return
    from trim.eval.model_tokenizer import load_model_encoding

    enc = load_model_encoding(str(model))
    assert enc.family == FAMILY_QWEN3
    ids = enc.build_first_turn_prompt_ids("When was Apple founded?")
    assert ids[0] == QWEN3_IM_START_ID or QWEN3_IM_START_ID in ids[:8]
    assert_qwen3_prompt_ids(ids)

