"""Contracts for zero-placebo and full-sampled-gap sampled-completion controls."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import textwrap

import pytest
import torch

from trim.adapters.components import zero_mask
from trim.cli.launch import LaunchError, parse_train_args
from trim.integrations.verl.batch_adapter import global_opd_weight_sum, plan_joint_sync_batches
from trim.integrations.verl.fsdp2_actor import unpack_gap_row, validate_gap_row_for_variant
from trim.state.snapshot import capture_snapshot
from trim.training.four_cell_runtime import build_manifest, cells_for_mode
from trim.training.opd_batch_health import assess_opd_health
from trim.training.opd_prompt_encoding import attach_prompt_context
from trim.training.opd_train_contract import (
    assert_train_contract,
    gap_variant_for_method,
    training_cell_for_method,
)
from trim.training.rl_opd_types import (
    COLLECTION_MODE_ZERO_PLACEBO,
    OPD_LOSS_CE,
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP,
    TRAINING_MODE_RL,
    TRAINING_MODE_SCAPE_SEED,
    HybridRolloutGroup,
    StudentDecisionPoint,
    collection_needs,
)
from trim.training.sr_opd_loss import gated_action_gap_weighted_sum, scale_gated_gap_loss, scale_projected_gap_loss
from trim.training.tinker_opd_datum import TinkerOPDDatum
from trim.training.tinker_rl_opd_trainer import hybrid_train_substep, prepare_hybrid_batch
from trim.training.train_checkpoint import assert_resume_identity


class _Enc:
    def build_first_turn_prompt_ids(self, query: str) -> list[int]:
        return [1]

    def build_continuation_prompt_ids(self, query: str, *, actions_obs: list, wm_text: str | None = None) -> list[int]:
        if "PRIVILEGED" in str(wm_text or ""):
            return [90, 91, 92]
        return [1, 2, 3]


def _snap(*, privileged: bool) -> object:
    snap = capture_snapshot(
        query_id="q0",
        step=0,
        harness_mask=zero_mask(),
        working_memory={"query": "Who founded the lab?"},
        query_text="Who founded the lab?",
    )
    attach_prompt_context(
        snap,
        acts=[],
        wm_text="student wm",
        teacher_wm_text="PRIVILEGED full harness evidence" if privileged else "",
    )
    return snap


def _point(
    *,
    episode: str = "q0_r0",
    turn: int = 0,
    student_ids: list[int] | None = None,
    action_ids: list[int] | None = None,
    teacher_ids: list[int] | None = None,
    valid: bool = True,
    privileged: bool = False,
) -> StudentDecisionPoint:
    snap = _snap(privileged=privileged)
    return StudentDecisionPoint(
        episode_id=episode,
        query_id="q0",
        rollout_idx=0,
        turn_id=turn,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="student",
        student_action_tokens=list(action_ids or [11, 12]),
        student_action_text="search",
        action_tool_names=["search_corpus"],
        reward=0.2,
        structurally_valid=valid,
        student_prompt_token_ids=list(student_ids or [1, 2, 3]),
        teacher_prompt_token_ids=list(teacher_ids or []),
    )


def _rl_row(point: StudentDecisionPoint, *, truncated: bool = False, valid: bool = True, prompt=None, actions=None) -> dict:
    return {
        "episode_id": point.episode_id,
        "turn_id": point.turn_id,
        "query_id": point.query_id,
        "effective_prompt_ids": list(point.student_prompt_token_ids if prompt is None else prompt),
        "action_ids": list(point.student_action_tokens if actions is None else actions),
        "valid": valid and not truncated,
        "truncated_generation": truncated,
        "finish_reason": "length" if truncated else "stop",
        "n_tokens": len(point.student_action_tokens),
    }


def _group(points: list[StudentDecisionPoint], rows: list[dict], rewards: list[float]) -> HybridRolloutGroup:
    return HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group={"rl_rows": rows},
        decision_points=points,
        terminal_rewards=rewards,
    )


def test_cli_and_cells_are_distinct_from_rl_and_trim():
    zero, _ = parse_train_args(["--train_method", "zero-placebo", "--component", "zero", "--out", "/tmp/zp"])
    full, _ = parse_train_args(["--train_method", "full-sampled-gap", "--component", "all", "--out", "/tmp/fsg"])
    assert training_cell_for_method(zero.train_method) != training_cell_for_method("rl")
    assert training_cell_for_method(full.train_method) != training_cell_for_method("trim")
    assert cells_for_mode(zero.training_mode, train_only=True) == ("zero_placebo",)
    assert cells_for_mode(full.training_mode, train_only=True) == ("full_sampled_gap",)
    zero_contract = assert_train_contract("zero-placebo", None, "torch_ddp_lora")
    full_contract = assert_train_contract("full-sampled-gap", None, "verl")
    assert zero_contract["gap_target_source"] == "sampled"
    assert full_contract["distributed_gap_supported"] is True
    assert full_contract["actor_objective"] == "gap"
    assert full_contract["control_variant"] == "full_sampled_gap"
    assert gap_variant_for_method("trim") == "projected"
    assert gap_variant_for_method("zero-placebo") == "sampled_zero_placebo"
    with pytest.raises(SystemExit):
        assert_train_contract("zero-placebo", OPD_LOSS_PROJECTED_GAP, "verl")
    with pytest.raises(SystemExit):
        assert_train_contract("full-sampled-gap", OPD_LOSS_CE, "verl")
    with pytest.raises(SystemExit):
        assert_train_contract("scape+rl", OPD_LOSS_SAMPLED_GAP, "verl")
    with pytest.raises(SystemExit):
        assert_train_contract("zero-placebo", None, "hf_debug")
    with pytest.raises(LaunchError):
        parse_train_args(["--train_method", "zero-placebo", "--harness", "Harness-G", "--component", "zero", "--out", "/tmp/g"])


def test_zero_placebo_copies_student_ids_and_skips_bad_rows():
    good = _point(teacher_ids=[8, 8, 8])
    bad_fmt = _point(episode="q0_r1", turn=1, valid=False)
    truncated = _point(episode="q0_r0", turn=2)
    rows = [
        _rl_row(good),
        _rl_row(bad_fmt),
        _rl_row(truncated, truncated=True),
    ]
    batch = prepare_hybrid_batch(
        groups=[_group([good, bad_fmt, truncated], rows, [0.2, 0.0])],
        rl_datums_by_query={"q0": rows},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="zero",
        teacher_event_fn=None,
        remove_constant_reward_groups=False,
        opd_states_per_trajectory=-1,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        control_variant="zero_placebo",
    )
    assert len(batch.opd_datums) == 1
    datum = batch.opd_datums[0]
    assert datum.prompt_token_ids == good.student_prompt_token_ids
    assert datum.teacher_prompt_token_ids == good.student_prompt_token_ids
    assert datum.target_tokens[-len(good.student_action_tokens) :] == good.student_action_tokens
    assert datum.metadata["sampled_action"] is True
    assert datum.metadata["projector_used"] is False
    assert datum.metadata["control_variant"] == "zero_placebo"
    assert datum.metadata["target_source"] == "student_on_policy_completion"
    assert datum.metadata["context_identical"] is True
    assert batch.projection_stats["n_projected_training_steps"] == 0
    assert batch.projection_stats["n_unregistered"] == 0
    assert batch.projection_stats["n_sampled_opd_datums"] == 1
    assert collection_needs(collection_mode=COLLECTION_MODE_ZERO_PLACEBO, opd_loss=OPD_LOSS_SAMPLED_GAP).need_teacher_context is False


def test_full_sampled_gap_uses_privileged_prefix_and_does_not_fall_back():
    point = _point(privileged=True, student_ids=[1, 2, 3], action_ids=[11, 12])
    row = _rl_row(point)
    batch = prepare_hybrid_batch(
        groups=[_group([point], [row], [0.4, 0.1])],
        rl_datums_by_query={"q0": [row]},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="all",
        teacher_event_fn=None,
        remove_constant_reward_groups=False,
        opd_states_per_trajectory=-1,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        model_enc=_Enc(),
        control_variant="full_sampled_gap",
    )
    datum = batch.opd_datums[0]
    assert datum.teacher_prompt_token_ids == [90, 91, 92]
    assert datum.teacher_prompt_token_ids != datum.prompt_token_ids
    assert datum.target_tokens[-2:] == [11, 12]
    assert datum.metadata["teacher_context_kind"] == "full_same_state"
    assert datum.metadata["context_identical"] is False
    assert batch.projection_stats["n_context_different"] == 1

    missing = _point(privileged=False, teacher_ids=[])
    missing_row = _rl_row(missing)
    with pytest.raises(ValueError, match="teacher-prefix recovery"):
        prepare_hybrid_batch(
            groups=[_group([missing], [missing_row], [0.4, 0.1])],
            rl_datums_by_query={"q0": [missing_row]},
            policy_version="v0",
            lambda_opd=0.01,
            component_id="all",
            teacher_event_fn=None,
            remove_constant_reward_groups=False,
            opd_states_per_trajectory=-1,
            opd_loss=OPD_LOSS_SAMPLED_GAP,
            model_enc=None,
            control_variant="full_sampled_gap",
        )


def test_variant_validation_rejects_cross_wiring():
    raw = {
        "prompt_ids": [1, 2],
        "target_ids": [3],
        "teacher_prompt_token_ids": [4, 5],
        "weights": [1.0],
        "opd_loss": OPD_LOSS_SAMPLED_GAP,
        "metadata": {"projector_used": False, "sampled_action": True, "control_variant": "full_sampled_gap"},
    }
    row = unpack_gap_row(raw, require_projected=False)
    validate_gap_row_for_variant(row, gap_variant="sampled_full")
    with pytest.raises(ValueError, match="projector_used"):
        unpack_gap_row(
            {
                **raw,
                "opd_loss": OPD_LOSS_PROJECTED_GAP,
                "metadata": {"projector_used": False, "sampled_action": True},
            },
            require_projected=True,
        )
    with pytest.raises(ValueError, match="projector_used"):
        validate_gap_row_for_variant(
            {**row, "opd_loss": OPD_LOSS_SAMPLED_GAP, "projector_used": True, "sampled_action": False},
            gap_variant="sampled_full",
        )
    with pytest.raises(ValueError, match="control_variant"):
        validate_gap_row_for_variant(row, gap_variant="sampled_zero_placebo")


def test_sampled_gap_gradient_matches_seed_formula():
    student = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    weights = torch.ones(2, dtype=torch.float32)
    numer = gated_action_gap_weighted_sum(student, teacher, weights, gate_beta=5.0)
    loss = scale_gated_gap_loss(numer, lambda_opd=0.01, z_gap=2.0, world_size=1)
    assert float(loss.detach()) == pytest.approx(0.0)
    loss.backward()
    assert student.grad is not None
    assert torch.allclose(student.grad, torch.tensor([-0.0025, -0.0025]))
    assert teacher.grad is None
    same = scale_projected_gap_loss(numer.detach(), lambda_opd=0.01, z_gap=2.0, world_size=1)
    assert float(same) == pytest.approx(0.0)

    student2 = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    teacher2 = torch.tensor([1.0, -0.5], dtype=torch.float32)
    numer2 = gated_action_gap_weighted_sum(student2, teacher2, weights, gate_beta=5.0)
    loss2 = scale_gated_gap_loss(numer2, lambda_opd=0.01, z_gap=2.0, world_size=1)
    loss2.backward()
    assert not torch.allclose(student2.grad, student.grad)

    delta = (teacher2 - student2.detach())
    gate = torch.sigmoid(delta * 5.0)
    assert torch.allclose(gate, torch.full((2,), 0.5)) is False


def test_identical_prefix_gate_is_one_half_without_old_logprobs():
    student = torch.tensor([0.2, -0.4], dtype=torch.float32, requires_grad=True)
    teacher = student.detach().clone()
    weights = torch.tensor([1.0, 1.0])
    numer = gated_action_gap_weighted_sum(student, teacher, weights, gate_beta=5.0)
    loss = scale_projected_gap_loss(numer, lambda_opd=0.01, z_gap=2.0, world_size=1)
    assert float(loss.detach()) == pytest.approx(0.0)
    loss.backward()
    assert torch.allclose(student.grad, torch.tensor([-0.0025, -0.0025]))


def test_z_gap_and_world_size_scale():
    datum = TinkerOPDDatum(
        model_input="p",
        prompt_token_ids=[1, 2],
        target_tokens=[0, 0, 9, 8],
        weights=[0.0, 0.0, 1.0, 1.0],
        policy_version="v0",
        n_supervised_tokens=2,
        teacher_prompt_token_ids=[3, 4],
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        metadata={"projector_used": False, "sampled_action": True, "control_variant": "zero_placebo"},
    )
    dummy = {"is_dummy": True, "weights": [5.0], "opd_loss": OPD_LOSS_SAMPLED_GAP}
    assert global_opd_weight_sum([datum, dummy]) == pytest.approx(2.0)
    one = scale_gated_gap_loss(torch.tensor(2.0), lambda_opd=0.01, z_gap=2.0, world_size=1)
    two = scale_gated_gap_loss(torch.tensor(2.0), lambda_opd=0.01, z_gap=2.0, world_size=2)
    assert float(two) == pytest.approx(float(one) * 2.0)
    assert float(two) > 0.0
    plan0 = plan_joint_sync_batches([], [datum], rank=0, world_size=2, micro_batch_size=1)
    plan1 = plan_joint_sync_batches([], [datum], rank=1, world_size=2, micro_batch_size=1)
    assert plan0.n_sync_rounds == plan1.n_sync_rounds
    assert plan0.global_opd_weight == pytest.approx(plan1.global_opd_weight)


def test_two_rank_gloo_gap_step_does_not_hang(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    script = tmp_path / "gloo_gap_worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import os, sys
            import torch
            from trim.training.sr_opd_loss import gated_action_gap_weighted_sum, scale_gated_gap_loss

            rank = int(sys.argv[1])
            port = sys.argv[2]
            out = sys.argv[3]
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = port
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = "2"
            torch.distributed.init_process_group("gloo", rank=rank, world_size=2)
            try:
                param = torch.nn.Parameter(torch.tensor([float(rank)], dtype=torch.float32))
                teacher = torch.zeros(1, dtype=torch.float32)
                numer = gated_action_gap_weighted_sum(param, teacher, torch.ones(1), gate_beta=5.0)
                loss = scale_gated_gap_loss(numer, lambda_opd=0.01, z_gap=2.0, world_size=2)
                loss.backward()
                grad = param.grad.detach().clone()
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.barrier()
                open(out, "w", encoding="utf-8").write(f"{rank} 1 {float(grad)}\\n")
            finally:
                torch.distributed.destroy_process_group()
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in [os.getcwd(), env.get("PYTHONPATH", "")] if p)
    procs = []
    outs = []
    for rank in range(2):
        out = tmp_path / f"rank{rank}.txt"
        outs.append(out)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(script), str(rank), str(port), str(out)],
                env=env,
                cwd=os.getcwd(),
            )
        )
    codes = [proc.wait(timeout=40) for proc in procs]
    assert codes == [0, 0]
    rows = [path.read_text(encoding="utf-8").split() for path in outs]
    assert [row[0] for row in rows] == ["0", "1"]
    assert [row[1] for row in rows] == ["1", "1"]


class _Spy:
    def __init__(self) -> None:
        self.calls: list = []

    def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        self.calls.append(("fb", loss_fn, len(list(data))))
        return {"loss": 0.1}

    def optim_step_async(self, adam, skip=False):
        self.calls.append(("opt", bool(skip)))


def test_joint_batch_takes_one_optimizer_step_and_keeps_opd_after_constant_rl():
    point = _point()
    row = _rl_row(point)
    batch = prepare_hybrid_batch(
        groups=[_group([point], [row], [1.0, 1.0])],
        rl_datums_by_query={"q0": [row]},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="zero",
        teacher_event_fn=None,
        remove_constant_reward_groups=True,
        opd_states_per_trajectory=-1,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        control_variant="zero_placebo",
    )
    assert batch.rl_datums == []
    assert batch.opd_datums
    assert batch.update_type.endswith("opd_only_zero_rl_signal") or "opd" in batch.update_type
    spy = _Spy()
    metrics = asyncio.run(
        hybrid_train_substep(
            training_client=spy,
            rl_datums=[{"n_tokens": 2, "target_tokens": [1, 2]}],
            opd_datums=batch.opd_datums,
            rl_loss_fn="cispo",
            rl_loss_fn_config={},
            lambda_opd=0.01,
            adam_params={},
            policy_version="v0",
            opd_loss=OPD_LOSS_SAMPLED_GAP,
        )
    )
    assert metrics.n_optimizer_steps == 1
    assert [call for call in spy.calls if call[0] == "opt"] == [("opt", False)]
    assert sum(1 for call in spy.calls if call[0] == "fb") == 2


def test_empty_opd_streak_and_positive_candidate_fail_closed():
    stats = {"n_positive_action_candidates": 0, "n_projected_training_steps": 0, "n_sampled_opd_datums": 0}
    first = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        n_rl=0,
        n_opd=0,
        projection_stats=stats,
        opd_empty_streak=0,
        max_empty_opd=3,
        control_variant="zero_placebo",
    )
    assert first["fatal"] is False
    second = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        n_rl=0,
        n_opd=0,
        projection_stats=stats,
        opd_empty_streak=first["opd_empty_streak"],
        max_empty_opd=3,
        control_variant="zero_placebo",
    )
    third = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        n_rl=0,
        n_opd=0,
        projection_stats=stats,
        opd_empty_streak=second["opd_empty_streak"],
        max_empty_opd=3,
        control_variant="zero_placebo",
    )
    assert third["fatal"] is True
    mismatched = _point()
    row = _rl_row(mismatched, prompt=[9, 9, 9])
    batch = prepare_hybrid_batch(
        groups=[_group([mismatched], [row], [0.2, 0.0])],
        rl_datums_by_query={"q0": [row]},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="zero",
        teacher_event_fn=None,
        remove_constant_reward_groups=False,
        opd_states_per_trajectory=-1,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        control_variant="zero_placebo",
    )
    health = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        n_rl=1,
        n_opd=len(batch.opd_datums),
        projection_stats=batch.projection_stats,
        opd_empty_streak=0,
        max_empty_opd=3,
        control_variant="zero_placebo",
    )
    assert health["fatal"] is True


def test_manifest_and_resume_stay_on_separate_cells():
    zero_args, _ = parse_train_args(
        ["--train_method", "zero-placebo", "--component", "zero", "--out", "/tmp/zp-man"]
    )
    manifest = build_manifest(zero_args)
    assert manifest["gap_target_source"] == "sampled"
    assert manifest["target_source"] == "student_on_policy_completion"
    assert manifest["protocol_name"] != "sr_projected_rl_opd"
    assert manifest["protocol_complete_rl_opd"] is False
    assert manifest["train_method"] == "zero-placebo"
    assert_resume_identity(
        {
            "train_method": "zero-placebo",
            "cell": "zero_placebo",
            "opd_loss": OPD_LOSS_SAMPLED_GAP,
            "component": "zero",
            "gap_variant": "sampled_zero_placebo",
        },
        train_method="zero-placebo",
        cell="zero_placebo",
        opd_loss=OPD_LOSS_SAMPLED_GAP,
        component="zero",
        gap_variant="sampled_zero_placebo",
    )
    with pytest.raises(SystemExit, match="cross-method"):
        assert_resume_identity(
            {
                "train_method": "trim",
                "cell": "scape_seed",
                "opd_loss": OPD_LOSS_PROJECTED_GAP,
                "component": "all",
                "gap_variant": "projected",
            },
            train_method="zero-placebo",
            cell="zero_placebo",
            opd_loss=OPD_LOSS_SAMPLED_GAP,
            component="zero",
            gap_variant="sampled_zero_placebo",
        )
    trim_args, _ = parse_train_args(["--train_method", "trim", "--component", "all", "--out", "/tmp/trim-man"])
    trim_manifest = build_manifest(trim_args)
    assert trim_manifest["training_mode"] == TRAINING_MODE_SCAPE_SEED
    assert trim_manifest["gap_target_source"] == "projected"
    rl_args, _ = parse_train_args(["--train_method", "rl", "--component", "all", "--out", "/tmp/rl-man"])
    assert build_manifest(rl_args)["training_mode"] == TRAINING_MODE_RL
