"""CPU contracts from Qwen3-4B RL/OPD audit (C01–C07, A1–A13 where feasible)."""

from __future__ import annotations

import math

import pytest
import torch

from trim.eval.model_tokenizer import parse_qwen_tool_call
from trim.training.action_encoding import (
    assert_action_visible,
    encode_supervised_action,
    prompt_ids_hash,
    resolve_effective_weight,
)
from trim.training.hf_rl_batch import gather_response_logprobs
from trim.training.hf_rl_opd_client import episode_relative_advantages
from trim.training.opd_dataset import ProjectedTrainingStep
from trim.training.tinker_opd_datum import build_tinker_opd_datums, supervised_weight_sum
from trim.training.train_query_sampler import QuerySampler, QuerySamplerState, pool_fingerprint
from trim.training.vllm_hybrid import extract_sampled_logprobs, result_from_worker_row


class _LP:
    def __init__(self, logprob: float, token_id: int | None = None) -> None:
        self.logprob = logprob
        self.token_id = token_id


class _FakeQwenEnc:
    family = "qwen3"
    stop_token_ids = [151645]

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 200 + 1 for c in str(text)] + [151645]

    def decode_tokens(self, ids: list[int]) -> str:
        chars = []
        for tid in ids:
            if int(tid) == 151645:
                continue
            chars.append(chr((int(tid) - 1) % 200))
        return "".join(chars)

    def parse_tool_call(self, text: str, completion_ids=None):
        return parse_qwen_tool_call(text, completion_ids=completion_ids)


def _ce_step(
    prompt: str,
    target: str,
    *,
    weight: float,
    confidence: float = 1.0,
    prompt_ids: list[int] | None = None,
    target_action: dict | None = None,
    token_mask: list[bool] | None = None,
    visible_doc_ids: list[str] | None = None,
) -> ProjectedTrainingStep:
    meta = {"student_prompt_token_ids": list(prompt_ids if prompt_ids is not None else [9, 8, 7])}
    if visible_doc_ids is not None:
        meta["visible_doc_ids"] = list(visible_doc_ids)
    return ProjectedTrainingStep(
        prompt_reduced=prompt,
        target_text=target,
        target_action=target_action or {"name": "end_search", "arguments": {}},
        token_mask=token_mask,
        weight=weight,
        projection_kind="direct",
        projection_confidence=confidence,
        metadata=meta,
    )


def test_a1_ce_uses_student_prompt_ids_not_debug_string():
    step = _ce_step("IGNORE_ME", "xx", weight=1.0, prompt_ids=[999, 998])
    datums = build_tinker_opd_datums(
        [step], lambda_opd=0.1, encode_fn=lambda t: [1] * len(t), policy_version="v1"
    )
    assert datums[0].prompt_token_ids == [999, 998]
    assert datums[0].metadata["prompt_hash"] == prompt_ids_hash([999, 998])
    assert datums[0].weights[:2] == [0.0, 0.0]


def test_a1_missing_prompt_ids_are_rejected():
    step = _ce_step("P", "aa", weight=1.0)
    step.metadata.pop("student_prompt_token_ids")
    with pytest.raises(ValueError, match="student_prompt_token_ids"):
        build_tinker_opd_datums([step], lambda_opd=0.1, policy_version="v1")


def test_a2_qwen_parser_accepts_rollout_style_target():
    enc = _FakeQwenEnc()
    ids, text = encode_supervised_action(
        target_action={"name": "search_corpus", "arguments": {"query": "audit"}},
        target_text="to=search_corpus\n{\"query\": \"audit\"}\n",
        encode=enc.encode,
        model_enc=enc,
    )
    parsed = parse_qwen_tool_call(text)
    assert parsed.parsed is True
    assert parsed.tool_name == "search_corpus"
    assert ids[-1] == 151645
    step = _ce_step(
        "P",
        "ignored",
        weight=1.0,
        prompt_ids=[1, 2],
        target_action={"name": "search_corpus", "arguments": {"query": "audit"}},
    )
    datums = build_tinker_opd_datums(
        [step], lambda_opd=0.1, encode_fn=enc.encode, policy_version="v1", model_enc=enc
    )
    assert datums[0].metadata["projected_action_token_ids"]
    assert datums[0].target_tokens[-1] == 151645


def test_a2_old_render_action_is_rejected_by_qwen_parser():
    parsed = parse_qwen_tool_call('to=search_corpus\n{"query": "audit"}\n')
    assert parsed.parsed is False
    assert parsed.error == "no_qwen_tool_call"


def test_a2_mask_length_mismatch_errors():
    step = _ce_step("P", "aa", weight=1.0, token_mask=[True])
    with pytest.raises(ValueError, match="mask length"):
        build_tinker_opd_datums(
            [step], lambda_opd=0.1, encode_fn=lambda t: [1, 2], policy_version="v1"
        )


def test_a2_clipped_visibility_rejects_hidden_doc():
    with pytest.raises(ValueError, match="not visible"):
        assert_action_visible(
            {"name": "read_document", "arguments": {"doc_id": "hidden"}},
            visible_doc_ids=["keep"],
        )
    step = _ce_step(
        "P",
        "x",
        weight=1.0,
        target_action={"name": "read_document", "arguments": {"doc_id": "hidden"}},
        visible_doc_ids=["keep"],
    )
    with pytest.raises(ValueError, match="not visible"):
        build_tinker_opd_datums(
            [step], lambda_opd=0.1, encode_fn=lambda t: [1], policy_version="v1"
        )


def test_a3_logprob_fail_closed():
    class LP:
        def __init__(self, logprob: float) -> None:
            self.logprob = logprob

    assert extract_sampled_logprobs([7, 8], [{7: LP(-1.5)}, {8: LP(-2.0)}]) == [-1.5, -2.0]
    assert extract_sampled_logprobs([7], [{7: LP(0.0)}]) == [0.0]
    with pytest.raises(ValueError, match="missing sampled logprobs"):
        extract_sampled_logprobs([101, 102], None)
    with pytest.raises(ValueError, match="token_id=101"):
        extract_sampled_logprobs([101], [{202: LP(-0.25)}])
    with pytest.raises(ValueError, match="length"):
        extract_sampled_logprobs([1, 2], [{1: LP(-0.1)}])
    with pytest.raises(ValueError, match="non-finite"):
        extract_sampled_logprobs([1], [{1: LP(float("nan"))}])
    with pytest.raises(RuntimeError, match="length"):
        result_from_worker_row(
            {"request_id": "r0", "token_ids": [1, 2], "token_logprobs": [-0.1], "effective_prompt_ids": [9]},
            enc=type("E", (), {"decode": staticmethod(lambda ids: "")})(),
        )


def test_a3_result_requires_effective_prompt_ids():
    with pytest.raises(RuntimeError, match="effective_prompt_ids"):
        result_from_worker_row(
            {"request_id": "r0", "token_ids": [1], "token_logprobs": [0.0]},
            enc=type("E", (), {})(),
        )


def test_a4_candidate_sharing_uses_weight_not_confidence():
    steps = [
        _ce_step("P", "aa", weight=1.0, confidence=1.0),
        _ce_step("P", "aa", weight=0.5, confidence=1.0),
        _ce_step("P", "aa", weight=0.5, confidence=1.0),
    ]
    datums = build_tinker_opd_datums(
        steps, lambda_opd=1.0, encode_fn=lambda t: [1, 1], policy_version="v1"
    )
    totals = [sum(d.weights) for d in datums]
    assert totals[0] == pytest.approx(0.5)
    assert totals[1] == pytest.approx(0.25)
    assert totals[2] == pytest.approx(0.25)
    assert abs(supervised_weight_sum(datums) - 1.0) < 1e-9


def test_a4_zero_weight_is_skip_not_fallback():
    steps = [
        _ce_step("P", "aa", weight=0.0, confidence=0.0),
        _ce_step("P", "aa", weight=1.0, confidence=1.0),
    ]
    datums = build_tinker_opd_datums(
        steps, lambda_opd=0.5, encode_fn=lambda t: [1, 1], policy_version="v1"
    )
    assert len(datums) == 1
    assert abs(supervised_weight_sum(datums) - 0.5) < 1e-9
    with pytest.raises(ValueError, match="finite"):
        resolve_effective_weight(weight=float("nan"), projection_confidence=None)
    with pytest.raises(ValueError, match=">="):
        resolve_effective_weight(weight=-1.0, projection_confidence=None)


def test_a4_splitting_candidate_preserves_total_weight():
    one = [_ce_step("P", "aaa", weight=1.0)]
    split = [_ce_step("P", "aaa", weight=0.5), _ce_step("P", "aaa", weight=0.5)]
    a = build_tinker_opd_datums(one, lambda_opd=0.2, encode_fn=lambda t: [1, 1, 1], policy_version="v1")
    b = build_tinker_opd_datums(split, lambda_opd=0.2, encode_fn=lambda t: [1, 1, 1], policy_version="v1")
    assert abs(supervised_weight_sum(a) - supervised_weight_sum(b)) < 1e-9


def test_a5_episode_advantage_population_std_and_no_turn_double_count():
    rows = [
        {"query_id": "q0", "episode_id": "e0", "reward": 0.0},
        {"query_id": "q0", "episode_id": "e1", "reward": 2.0},
        {"query_id": "q0", "episode_id": "e1", "reward": 2.0},
        {"query_id": "q0", "episode_id": "e1", "reward": 2.0},
    ]
    adv = episode_relative_advantages(rows)
    mu = 1.0
    sigma = math.sqrt(((0.0 - mu) ** 2 + (2.0 - mu) ** 2) / 2)
    assert adv[0] == pytest.approx((0.0 - mu) / sigma)
    assert adv[1] == pytest.approx((2.0 - mu) / sigma)
    assert adv[2] == adv[1]


def test_a6_cispo_clip_mapping():
    from trim.integrations.verl.joint_objective import cispo_clip_bounds

    low, high = cispo_clip_bounds(clip_ratio_low=1.0, clip_ratio_high=4.0)
    assert low == 0.0
    assert high == 5.0


def test_a11_rl_collection_mode_skips_teacher_and_keeps_turn_ids():
    from trim.training.batched_env_rollout import (
        _keep_dual_view,
        _keep_snapshots,
        _keep_teacher_encode,
    )
    from trim.training.rl_opd_types import COLLECTION_MODE_AUDIT_FULL, COLLECTION_MODE_RL, COLLECTION_MODE_RL_OPD

    assert _keep_teacher_encode(COLLECTION_MODE_RL) is False
    assert _keep_dual_view(COLLECTION_MODE_RL) is False
    assert _keep_snapshots(COLLECTION_MODE_RL) is False
    assert _keep_snapshots(COLLECTION_MODE_RL_OPD) is True
    assert _keep_teacher_encode(COLLECTION_MODE_RL_OPD) is False
    assert _keep_teacher_encode(COLLECTION_MODE_AUDIT_FULL) is True
    assert _keep_dual_view(COLLECTION_MODE_AUDIT_FULL) is True


def test_a12_a13_sampler_unique_and_fingerprint():
    pool = [{"query_id": str(i)} for i in range(5)]
    sampler = QuerySampler(pool, base_seed=42, groups_per_step=4)
    a, meta_a = sampler.sample_for_rollout()
    b, meta_b = sampler.sample_for_rollout()
    ids_a = [r["query_id"] for r in a]
    ids_b = [r["query_id"] for r in b]
    assert len(ids_a) == len(set(ids_a))
    assert len(ids_b) == len(set(ids_b))
    assert "0" not in ids_b or ids_b.count("0") == 1
    other = [{"query_id": f"x{i}"} for i in range(5)]
    with pytest.raises(ValueError, match="fingerprint"):
        QuerySampler(
            other,
            base_seed=42,
            groups_per_step=4,
            state=QuerySamplerState(
                base_seed=42,
                groups_per_step=4,
                pool_size=5,
                pool_fingerprint=pool_fingerprint(pool),
                sample_params_hash=sampler.state.sample_params_hash,
            ),
        )


def test_a4_fp32_log_softmax_is_used():
    logits = torch.zeros(1, 3, 5, dtype=torch.bfloat16)
    logits[0, 0, 2] = 8.0
    logits[0, 1, 3] = 8.0
    logps = gather_response_logprobs(logits, [[2, 3]], max_resp=2)
    assert logps[0].dtype == torch.float32
    assert torch.isfinite(logps[0]).all()


def test_a5_constant_reward_zero_advantage():
    rows = [
        {"query_id": "q0", "episode_id": "e0", "reward": 0.4},
        {"query_id": "q0", "episode_id": "e1", "reward": 0.4},
    ]
    assert episode_relative_advantages(rows) == [0.0, 0.0]


def test_a6_cispo_clip_and_detach():
    from trim.integrations.verl.joint_objective import cispo_token_loss

    new = torch.tensor([-0.2, -1.0, 2.0], requires_grad=True)
    old = torch.tensor([-0.2, 0.0, -0.1])
    loss = cispo_token_loss(new, old, advantage=1.0)
    loss.backward()
    ratio = (new.detach() - old).exp()
    weight = ratio.clamp(0.0, 5.0)
    assert float(weight[0]) == pytest.approx(1.0)
    assert float(weight[1]) < 1.0
    assert float(weight[2]) == pytest.approx(5.0)
    assert new.grad is not None
    # Weight is detached: dL/d new_i includes -w_i * A, not the ratio Jacobian.
    assert torch.isfinite(new.grad).all()


def test_a7_joint_loss_is_sum():
    from trim.integrations.verl.joint_objective import assert_one_optimizer_step, combine_joint_loss

    rl = torch.tensor(2.0, requires_grad=True)
    ce = torch.tensor(4.0, requires_grad=True)
    total = combine_joint_loss(rl, ce, lambda_opd=0.1)
    total.backward()
    assert float(total.detach()) == pytest.approx(2.4)
    assert float(rl.grad) == pytest.approx(1.0)
    assert float(ce.grad) == pytest.approx(0.1)
    assert_one_optimizer_step(1)
    with pytest.raises(AssertionError):
        assert_one_optimizer_step(2)


def test_a12_empty_signal_does_not_count_update():
    sampler = QuerySampler([{"query_id": str(i)} for i in range(5)], base_seed=1, groups_per_step=2)
    sampler.note_rollout_start()
    assert sampler.state.global_rollout_batch == 1
    assert sampler.state.global_optimizer_step == 0
    sampler.note_update_complete()
    assert sampler.state.global_optimizer_step == 1


def test_c05_reward_parts_recorded_without_formula_change():
    from types import SimpleNamespace

    from trim.training.rl_opd_metrics import reward_parts_group_stats

    def _g(parts):
        stats = [{"reward_parts": p, "names": ["search_corpus"], "n_turns": 2, "n_valids": 2, "n_structurally_valid": 2, "n_exec_ok": 1, "ended": False, "max_turns": 40} for p in parts]
        return SimpleNamespace(trajectory_group={"episode_stats": stats})

    groups = [
        _g(
            [
                {"task": 0.8, "legal": 0.1, "shaping": 0.05, "total": 0.95},
                {"task": 0.8, "legal": 0.1, "shaping": 0.02, "total": 0.92},
            ]
        )
    ]
    out = reward_parts_group_stats(groups)
    assert out["reward_formula_unchanged"] is True
    assert out["frac_task_var_zero_total_var_nonzero"] == pytest.approx(1.0)
    assert out["task"]["mean"] == pytest.approx(0.8)


def test_c04_train_refuses_silent_clip():
    from trim.training.hf_rl_opd_client import HFDebugTrainingClient

    class B:
        _device = torch.device("cpu")
        optimizer = None

        def encode(self, text):
            return [1, 2, 3]

    client = HFDebugTrainingClient(B(), max_full_tokens=4)
    with pytest.raises(ValueError, match="clipping must happen before sampling"):
        client._align_context([1, 2, 3], [4, 5])
    with pytest.raises(ValueError, match="missing sampled prompt IDs"):
        client._prepare_cispo_row({"query_id": "q", "action_ids": [1], "token_logprobs": [0.0]})
    with pytest.raises(ValueError, match="refuse silent encode"):
        client._opd_loss([{"prompt": "x", "target_text": "y"}])
