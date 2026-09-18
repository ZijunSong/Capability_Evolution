"""CPU contracts for trim / projected-gap multi-GPU (2026-09-18 plan)."""

from __future__ import annotations

import json

import pytest
import torch

from trim.integrations.verl.batch_adapter import (
    dummy_opd_row,
    global_opd_weight_sum,
    plan_joint_sync_batches,
)
from trim.integrations.verl.fsdp2_actor import detect_opd_objective, unpack_gap_row
from trim.training.opd_train_contract import (
    assert_train_contract,
    normalize_opd_loss,
    training_cell_for_method,
)
from trim.training.sr_opd_loss import (
    gated_action_gap_per_token,
    gated_action_gap_weighted_sum,
    scale_projected_gap_loss,
)
from trim.training.tinker_opd_datum import TinkerOPDDatum


def test_trim_defaults_to_projected_gap_not_ce():
    assert normalize_opd_loss(None, method="trim") == "sr_opd_projected_gap"
    assert normalize_opd_loss("", method="scape_seed") == "sr_opd_projected_gap"
    assert normalize_opd_loss(None, method="rl+opd") == "sr_opd_ce"
    assert training_cell_for_method("trim") == "scape_seed"
    assert training_cell_for_method("rl+opd") == "rl_opd"
    assert training_cell_for_method("rl") == "rl"


def test_trim_gap_backends_are_supported():
    for backend in ("hf_debug", "verl", "fsdp2", "verl_fsdp2", "torch_ddp_lora"):
        payload = assert_train_contract("trim", None, backend)
        assert payload["opd_loss"] == "sr_opd_projected_gap"
        assert payload["actor_objective"] == "gap"
        assert payload["supported"] is True
    ddp = assert_train_contract("trim", "sr_opd_projected_gap", "torch_ddp_lora")
    assert ddp["backend"] == "torch_ddp_lora"
    assert ddp["actor_wrap"] == "ddp"
    fsdp = assert_train_contract("trim", "sr_opd_projected_gap", "verl")
    assert fsdp["actor_wrap"] == "fsdp2"


def test_equal_logprobs_have_half_gate_student_grad():
    student = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([0.0, 0.0], dtype=torch.float32)
    weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    numer = gated_action_gap_weighted_sum(student, teacher, weights, gate_beta=5.0)
    assert float(numer.detach()) == pytest.approx(0.0)
    lam = 0.01
    z_gap = 2.0
    loss = scale_projected_gap_loss(numer, lambda_opd=lam, z_gap=z_gap, world_size=1)
    loss.backward()
    # g=σ(0)=1/2; dL/dℓ^S = −λ w g / Z = −0.01 * 1 * 0.5 / 2
    assert student.grad is not None
    assert torch.allclose(student.grad, torch.tensor([-0.0025, -0.0025]))
    assert teacher.grad is None


def test_teacher_logprob_change_moves_student_grad():
    student = torch.tensor([0.2, -0.1], dtype=torch.float32, requires_grad=True)
    teacher_a = torch.tensor([0.5, 0.0], dtype=torch.float32)
    teacher_b = torch.tensor([-0.5, 1.0], dtype=torch.float32)
    weights = torch.tensor([0.4, 1.6], dtype=torch.float32)
    numer_a = gated_action_gap_weighted_sum(student, teacher_a, weights, gate_beta=5.0)
    loss_a = scale_projected_gap_loss(numer_a, lambda_opd=0.01, z_gap=2.0, world_size=1)
    loss_a.backward()
    grad_a = student.grad.detach().clone()
    student.grad = None
    numer_b = gated_action_gap_weighted_sum(student, teacher_b, weights, gate_beta=5.0)
    loss_b = scale_projected_gap_loss(numer_b, lambda_opd=0.01, z_gap=2.0, world_size=1)
    loss_b.backward()
    assert not torch.allclose(grad_a, student.grad)


def test_gap_helper_rejects_length_mismatch():
    student = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([0.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="length mismatch"):
        gated_action_gap_per_token(student, teacher, gate_beta=5.0)


def test_zero_z_gap_skips_without_clamp():
    numer = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
    loss = scale_projected_gap_loss(numer, lambda_opd=0.01, z_gap=0.0, world_size=8)
    assert float(loss.detach()) == pytest.approx(0.0)
    loss.backward()
    assert float(numer.grad) == pytest.approx(0.0)


def test_world_size_scale_undoes_mean_reduction():
    numer = torch.tensor(2.0, dtype=torch.float32)
    one = scale_projected_gap_loss(numer, lambda_opd=0.01, z_gap=4.0, world_size=1)
    eight = scale_projected_gap_loss(numer, lambda_opd=0.01, z_gap=4.0, world_size=8)
    assert float(eight) == pytest.approx(float(one) * 8.0)


def test_dummy_opd_row_has_teacher_and_gap_loss():
    row = dummy_opd_row(opd_loss="sr_opd_projected_gap")
    assert row["is_dummy"] is True
    assert row["teacher_prompt_token_ids"]
    assert row["opd_loss"] == "sr_opd_projected_gap"
    assert row["weights"] == [0.0]
    unpacked = unpack_gap_row(row, require_projected=False)
    assert unpacked["dummy"] is True
    assert unpacked["teacher_ids"]
    assert unpacked["student_ids"]


def test_z_gap_from_global_list_excludes_dummy_and_prompt():
    datum = TinkerOPDDatum(
        model_input="p",
        prompt_token_ids=[1, 2],
        target_tokens=[0, 0, 9, 8],
        weights=[0.0, 0.0, 0.5, 1.5],
        policy_version="v0",
        n_supervised_tokens=2,
        teacher_prompt_token_ids=[3, 4, 5],
        opd_loss="sr_opd_projected_gap",
        metadata={"projector_used": True, "sampled_action": False, "lambda_opd": 0.01, "gate_beta": 5.0},
    )
    dummy = dummy_opd_row(opd_loss="sr_opd_projected_gap")
    z = global_opd_weight_sum([datum, dummy])
    assert z == pytest.approx(2.0)
    plan = plan_joint_sync_batches([], [datum], rank=0, world_size=8, micro_batch_size=1)
    assert plan.global_opd_weight == pytest.approx(2.0)
    assert plan.n_dummy_opd == 7
    assert detect_opd_objective([datum]) == "gap"


def test_datum_roundtrip_keeps_gap_fields():
    datum = TinkerOPDDatum(
        model_input="reduced",
        prompt_token_ids=[1, 2, 3],
        target_tokens=[0, 0, 0, 11, 12],
        weights=[0.0, 0.0, 0.0, 0.4, 0.0],
        policy_version="v7",
        n_supervised_tokens=1,
        projection_confidence=0.4,
        target_action={"name": "search_corpus", "arguments": {"query": "q"}},
        teacher_prompt_token_ids=[9, 8, 7, 6],
        opd_loss="sr_opd_projected_gap",
        metadata={
            "projector_used": True,
            "sampled_action": False,
            "target_source": "projected",
            "lambda_opd": 0.01,
            "gate_beta": 5.0,
            "decision_point_id": "ep:3",
            "context_hash": "abc",
        },
    )
    blob = json.loads(json.dumps(datum.to_dict()))
    restored = TinkerOPDDatum.from_dict(blob)
    assert restored.prompt_token_ids == datum.prompt_token_ids
    assert restored.teacher_prompt_token_ids == datum.teacher_prompt_token_ids
    assert restored.target_tokens == datum.target_tokens
    assert restored.weights == datum.weights
    assert restored.opd_loss == "sr_opd_projected_gap"
    assert restored.metadata["target_source"] == "projected"
    assert restored.metadata["decision_point_id"] == "ep:3"
    with pytest.raises(ValueError, match="missing loss_id"):
        TinkerOPDDatum.from_dict({"prompt_token_ids": [1], "target_tokens": [1], "weights": [1.0]})


def test_unpack_refuses_sampled_action_on_projected_path():
    raw = {
        "prompt_ids": [1, 2],
        "target_ids": [3],
        "teacher_prompt_token_ids": [4, 5],
        "weights": [1.0],
        "opd_loss": "sr_opd_projected_gap",
        "metadata": {"projector_used": False, "sampled_action": True},
    }
    with pytest.raises(ValueError, match="projector_used"):
        unpack_gap_row(raw, require_projected=True)


def test_teacher_encode_uses_wm_when_acts_empty():
    from trim.training.opd_prompt_encoding import encode_teacher_rollout_style_prompt

    class _FakeEnc:
        def build_first_turn_prompt_ids(self, query: str) -> list[int]:
            return [1]

        def build_continuation_prompt_ids(
            self,
            query: str,
            *,
            actions_obs: list,
            wm_text: str | None = None,
        ) -> list[int]:
            assert wm_text == "teacher wm"
            return [9, 8, 7]

    ids, _ = encode_teacher_rollout_style_prompt(_FakeEnc(), "Who?", acts=[], wm_text="teacher wm")
    assert ids == [9, 8, 7]
    ids_first, _ = encode_teacher_rollout_style_prompt(_FakeEnc(), "Who?", acts=[], wm_text="")
    assert ids_first == [1]
