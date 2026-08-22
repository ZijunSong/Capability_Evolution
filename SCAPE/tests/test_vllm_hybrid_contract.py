from __future__ import annotations

import pytest

from scape.eval.harmony_runtime import CANONICAL_STOP_TOKEN_IDS, O200K_HARMONY, stop_ids_for_tool_actions
from scape.training.vllm_hybrid import (
    GenerateResult,
    SchemeARuntime,
    assert_gptoss_tokenizer,
    cispo_row_from_generation,
    extract_sampled_logprobs,
    mean_behavior_logprob,
    plan_cell_phases,
)


class _GptOssTok:
    name_or_path = "/data/ppnm/models/gpt-oss-20b"
    vocab_size = 201088

    def convert_tokens_to_ids(self, tok: str) -> int:
        return {"<|call|>": 200012, "<|return|>": 200002}[tok]


class _Cl100kTok:
    name_or_path = "cl100k_base"
    vocab_size = 100277

    def convert_tokens_to_ids(self, tok: str) -> int:
        return 1


def test_harmony_stop_tokens_are_call_and_return_only():
    assert stop_ids_for_tool_actions() == [200012, 200002]
    assert CANONICAL_STOP_TOKEN_IDS == [200012, 200002]
    assert O200K_HARMONY == "o200k_harmony"


def test_plan_rl_opd_refreshes_every_optimizer_step():
    phases = plan_cell_phases("rl_opd", train_steps=8, on_policy_refresh=True, use_frozen_states=False)
    assert phases.count("vllm_rollout") == 8
    assert phases.count("hf_train") == 8
    assert phases[-1] == "vllm_eval"
    pairs = list(zip(phases, phases[1:]))
    assert ("vllm_rollout", "hf_train") in pairs
    assert ("hf_train", "vllm_rollout") in pairs


def test_plan_pure_opd_frozen_does_not_rerollout():
    phases = plan_cell_phases("pure_opd", train_steps=8, on_policy_refresh=True, use_frozen_states=True)
    assert "vllm_rollout" not in phases
    assert phases.count("hf_train") == 8
    assert phases[-1] == "vllm_eval"


def test_plan_before_is_eval_only():
    assert plan_cell_phases("before", train_steps=8, on_policy_refresh=True, use_frozen_states=False) == [
        "vllm_rollout",
        "vllm_eval",
    ]


def test_cispo_datum_keeps_vllm_token_ids_and_logprobs():
    gen = GenerateResult(
        request_id="q0:g0:t0:0",
        token_ids=[10, 11, 200012],
        token_logprobs=[-0.2, -0.4, -0.1],
        text="to=functions.search_corpus",
        logprob_old=mean_behavior_logprob([-0.2, -0.4, -0.1]),
        logprob_provenance="vllm_sampled_token",
    )
    row = cispo_row_from_generation(
        query_id="q0",
        prompt_ids=[1, 2, 3],
        prompt_text="prompt",
        gen=gen,
        policy_version="v3",
        turn_id=1,
        valid=True,
    )
    assert row["action_ids"] == [10, 11, 200012]
    assert row["token_logprobs"] == [-0.2, -0.4, -0.1]
    assert row["action_mask"] == [1, 1, 1]
    assert row["logprob_provenance"] == "vllm_sampled_token"
    assert row["policy_version"] == "v3"
    assert row["logprob_old"] == pytest.approx((-0.2 - 0.4 - 0.1) / 3)


def test_sampled_logprob_extractor_reads_vllm_dict():
    class LP:
        def __init__(self, logprob: float) -> None:
            self.logprob = logprob

    raw = [{7: LP(-1.5)}, {8: LP(-2.0)}]
    assert extract_sampled_logprobs([7, 8], raw) == [-1.5, -2.0]


def test_tokenizer_audit_accepts_gptoss_and_rejects_cl100k():
    audit = assert_gptoss_tokenizer(_GptOssTok(), source="/data/ppnm/models/gpt-oss-20b")
    assert audit["encoding"] == O200K_HARMONY
    assert audit["special_token_ids"]["<|call|>"] == 200012
    with pytest.raises(RuntimeError, match="cl100k"):
        assert_gptoss_tokenizer(_Cl100kTok(), source="cl100k_base")


def test_scheme_a_forbids_hf_and_vllm_resident_together():
    runtime = SchemeARuntime()

    class FakeVLLM:
        alive = True

        def close(self) -> None:
            self.alive = False

    runtime.attach_hf(object())
    with pytest.raises(RuntimeError, match="cannot start vLLM"):
        runtime.attach_vllm(FakeVLLM())  # type: ignore[arg-type]
    runtime.detach_hf()
    runtime.attach_vllm(FakeVLLM())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="cannot load HF"):
        runtime.attach_hf(object())
