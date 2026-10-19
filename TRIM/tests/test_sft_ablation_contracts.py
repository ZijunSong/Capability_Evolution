from __future__ import annotations

import json

import pytest

from trim.training.sft_ablation.cli import parse_sft_args
from trim.training.sft_ablation.contracts import (
    PROFILES,
    SFTBatchRow,
    profile_record,
    require_same_identity,
)
from trim.training.sft_ablation.dist_launch import launch_sft
from trim.training.sft_ablation.geometry import world_cards


def _row(experiment: str, **overrides):
    profile = PROFILES[experiment]
    payload = dict(
        row_id=f"{experiment}-1",
        experiment=experiment,
        query_id="q",
        episode_id="ep",
        decision_id="d",
        prompt_ids=[1, 2],
        response_ids=[3],
        response_mask=[True],
        weight=1.0,
        policy_version="p",
        prompt_owner=profile.prompt_owner,
        target_source=profile.supervision_provider,
    )
    payload.update(overrides)
    return SFTBatchRow(**payload)


def test_three_profiles_are_distinct_and_fixed():
    assert set(PROFILES) == {"teacher_sft", "projected_teacher_sft", "student_onpolicy_sft"}
    assert PROFILES["teacher_sft"].prompt_owner == "teacher_full"
    assert PROFILES["teacher_sft"].projection is False
    assert PROFILES["projected_teacher_sft"].prompt_owner == "student_zero"
    assert PROFILES["student_onpolicy_sft"].supervision_provider == "trim_component_branches_v1"
    assert PROFILES["student_onpolicy_sft"].online_refresh is True
    for profile in PROFILES.values():
        assert profile.rl_enabled is False
        assert profile.gap_enabled is False
        assert profile.gate_enabled is False
        assert profile.eval_harness == "zero"
        assert profile.teacher_harness == "all"


def test_rows_are_not_interchangeable():
    teacher = _row("teacher_sft")
    teacher.validate(32)
    projected = _row("projected_teacher_sft")
    projected.validate(32)
    swapped = teacher.to_dict()
    swapped.pop("schema_version")
    swapped["experiment"] = "projected_teacher_sft"
    with pytest.raises(ValueError, match="prompt owner"):
        SFTBatchRow(**swapped).validate(32)
    with pytest.raises(ValueError, match="target source"):
        _row("student_onpolicy_sft", target_source="full_model_trajectory_v1").validate(32)


def test_resume_refuses_cross_experiment():
    left = profile_record("teacher_sft")
    right = profile_record("student_onpolicy_sft")
    with pytest.raises(ValueError, match="resume refused"):
        require_same_identity(left, right)
    require_same_identity(left, dict(left))


def test_parsers_reject_rl_flags():
    base = ["--model_name", "m", "--out", "/tmp/sft-out"]
    for experiment in PROFILES:
        with pytest.raises(SystemExit, match="rl-reward"):
            parse_sft_args(experiment, base + ["--rl-reward", "final-answer-recall"])
        with pytest.raises(SystemExit, match="train_method"):
            parse_sft_args(experiment, base + ["--train_method", "trim"])
    with pytest.raises(SystemExit, match="teacher-model"):
        parse_sft_args("student_onpolicy_sft", base + ["--teacher-model", "m"])
    with pytest.raises(SystemExit):
        parse_sft_args("student_onpolicy_sft", base + ["--stage", "prepare"])


def test_dry_run_is_32_cards_and_does_not_call_the_worker(capsys):
    assert world_cards(4, 8, 1) == 32

    def worker(args, dist):
        raise AssertionError("dry-run started a worker")

    code = launch_sft(
        script=__import__("pathlib").Path(__file__),
        experiment="teacher_sft",
        worker=worker,
        argv=["--model_name", "m", "--out", "/tmp/teacher-sft", "--dry-run"],
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["world_cards"] == 32
    assert payload["nnodes"] == 4
    assert payload["rollout_replicas"] == 8
    assert payload["nproc_per_node"] == 1
    assert payload["gradient_ranks"] == 1
    assert payload["teacher_mask"] == "all"
    assert payload["student_mask"] == "zero"
    assert payload["rl_enabled"] is False
    assert payload["loss"] == "action_ce_v1"


def test_nproc_other_than_one_is_refused():
    with pytest.raises(SystemExit, match="nproc-per-node"):
        launch_sft(
            script=__import__("pathlib").Path(__file__),
            experiment="teacher_sft",
            worker=lambda args, dist: 0,
            argv=[
                "--nnodes",
                "1",
                "--nproc-per-node",
                "2",
                "--rollout-replicas",
                "1",
                "--model_name",
                "m",
                "--out",
                "/tmp/x",
                "--dry-run",
            ],
        )
