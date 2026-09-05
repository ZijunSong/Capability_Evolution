from __future__ import annotations

import pytest

from trim.eval.harmony_runtime import (
    HARMONY_START_ID,
    assert_o200k_harmony_token_ids,
    load_harmony_enc,
)
from trim.eval.harmony_hf_encoding import (
    TokenizerHarmonyEncoding,
    render_harmony_conversation_text,
)


ASCII_ROLE_SYSTEM = [
    91, 82, 111, 108, 101, 46, 83, 89, 83, 84, 69, 77, 93, 32, 123, 34, 99, 104, 97, 110
]


def test_rejects_character_fallback_prompt_ids():
    with pytest.raises(RuntimeError, match="character fallback"):
        assert_o200k_harmony_token_ids(ASCII_ROLE_SYSTEM, what="recovery5 prompt")


def test_rejects_low_ascii_prompt_ids():
    with pytest.raises(RuntimeError, match="character fallback"):
        assert_o200k_harmony_token_ids(list(range(32, 120)), what="ord() prompt")


def test_accepts_harmony_start_prompt():
    ids = [HARMONY_START_ID, 17360, 200008, 3575, 200007, HARMONY_START_ID]
    assert assert_o200k_harmony_token_ids(ids)[0] == HARMONY_START_ID


def test_load_harmony_enc_refuses_local_fallback(monkeypatch):
    import trim.eval.harmony_runtime as hr

    class _Fake:
        def stop_tokens_for_assistant_actions(self):
            return [200002, 200012]

    monkeypatch.setattr(hr, "_configure_tiktoken_offline_paths", lambda model_path=None: None)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no azure vocab")

    monkeypatch.setattr(
        "openai_harmony.load_harmony_encoding",
        _boom,
        raising=False,
    )
    # Force the rust import path to fail inside load_harmony_enc.
    def _failing_load(*_a, **_k):
        raise RuntimeError("no azure vocab")

    import openai_harmony

    monkeypatch.setattr(openai_harmony, "load_harmony_encoding", _failing_load)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        load_harmony_enc(model_path=None)


def test_load_harmony_enc_uses_checkpoint_tokenizer_when_rust_fails(monkeypatch):
    from pathlib import Path

    model = Path("/data/ppnm/models/harness-1")
    if not (model / "tokenizer.json").is_file():
        pytest.skip("gpt-oss tokenizer not on disk")

    import openai_harmony
    from trim.eval.harmony_hf_encoding import TokenizerHarmonyEncoding
    from trim.eval.harmony_runtime import build_first_turn_prompt_ids

    def _boom(*_a, **_k):
        raise RuntimeError("no azure vocab")

    monkeypatch.setattr(openai_harmony, "load_harmony_encoding", _boom)
    enc = load_harmony_enc(str(model))
    assert isinstance(enc, TokenizerHarmonyEncoding)
    ids = build_first_turn_prompt_ids("When was Apple founded?", enc=enc)
    assert ids[0] == HARMONY_START_ID


def test_hf_renderer_matches_rust_first_turn():
    from openai_harmony import Role
    from harness.ultra_core import build_context, get_system_prompt
    from trim.eval.harmony_runtime import decode_ids

    enc = load_harmony_enc()
    if type(enc).__name__ == "TokenizerHarmonyEncoding":
        pytest.skip("rust Harmony encoding unavailable")
    conv = build_context(get_system_prompt("When was Apple founded?"), None, [], [])
    gold = decode_ids(enc, enc.render_conversation_for_completion(conv, Role.ASSISTANT))
    ours = render_harmony_conversation_text(conv, next_role=Role.ASSISTANT)
    assert gold == ours
    ids = assert_o200k_harmony_token_ids(
        enc.render_conversation_for_completion(conv, Role.ASSISTANT)
    )
    assert ids[0] == HARMONY_START_ID


def test_tokenizer_encoding_matches_rust_when_model_present():
    from pathlib import Path
    from openai_harmony import Role
    from harness.ultra_core import build_context, get_system_prompt

    model = Path("/data/ppnm/models/harness-1")
    if not (model / "tokenizer.json").is_file():
        pytest.skip("gpt-oss tokenizer not on disk")
    rust = load_harmony_enc()
    if type(rust).__name__ == "TokenizerHarmonyEncoding":
        pytest.skip("rust Harmony encoding unavailable")
    hf = TokenizerHarmonyEncoding.from_pretrained(str(model))
    conv = build_context(get_system_prompt("When was Apple founded?"), None, [], [])
    gold = list(rust.render_conversation_for_completion(conv, Role.ASSISTANT))
    ours = hf.render_conversation_for_completion(conv, Role.ASSISTANT)
    assert ours == gold
    assert_o200k_harmony_token_ids(ours)
