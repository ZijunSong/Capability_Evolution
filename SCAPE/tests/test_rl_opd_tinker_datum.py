from __future__ import annotations

import pytest

from scape.training.opd_dataset import ProjectedTrainingStep
from scape.training.tinker_opd_datum import (
    build_sampled_opd_datums,
    build_tinker_opd_datums,
    supervised_weight_sum,
)


def _step(prompt: str, target: str, *, confidence: float = 1.0) -> ProjectedTrainingStep:
    return ProjectedTrainingStep(
        prompt_reduced=prompt,
        target_text=target,
        target_action={"name": "curate", "arguments": {}},
        token_mask=None,
        weight=confidence,
        projection_kind="direct",
        projection_confidence=confidence,
    )


def _encode(text: str) -> list[int]:
    return [1] * len(text)


def test_opd_weights_sum_to_lambda():
    steps = [_step("PREFIX", "aa"), _step("PREFIX", "bbb")]
    datums = build_tinker_opd_datums(
        steps, lambda_opd=0.5, encode_fn=_encode, policy_version="v1"
    )
    assert abs(supervised_weight_sum(datums) - 0.5) < 1e-9
    assert [d.n_supervised_tokens for d in datums] == [2, 3]


def test_prompt_token_weight_zero():
    datums = build_tinker_opd_datums(
        [_step("SYS USER STATE", "xx")],
        lambda_opd=1.0,
        encode_fn=_encode,
        policy_version="v1",
    )
    d = datums[0]
    n_prompt = len(_encode("SYS USER STATE"))
    assert all(w == 0.0 for w in d.weights[:n_prompt])
    assert all(w > 0.0 for w in d.weights[n_prompt:])


def test_lambda_zero_builds_no_datums():
    assert (
        build_tinker_opd_datums(
            [_step("P", "aa")], lambda_opd=0.0, encode_fn=_encode, policy_version="v1"
        )
        == []
    )


def test_teacher_marker_rejected_from_model_input():
    with pytest.raises(ValueError, match="leaked"):
        build_tinker_opd_datums(
            [_step("teacher_verify_judgment: YES", "aa")],
            lambda_opd=0.1,
            encode_fn=_encode,
            policy_version="v1",
        )


def test_sampled_opd_datums_use_action_tokens_not_encoded_text():
    from scape.adapters.components import minus_mask
    from scape.state.snapshot import capture_snapshot
    from scape.training.rl_opd_types import StudentDecisionPoint

    snap = capture_snapshot(
        query_id="q0",
        step=0,
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={"curated_ids": ["d1"], "accessible_doc_ids": ["d1"]},
        metadata={"component_id": "auto_populate_first_search"},
    )
    point = StudentDecisionPoint(
        episode_id="e0",
        query_id="q0",
        rollout_idx=0,
        turn_id=0,
        policy_version="v1",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="student-prefix",
        student_action_tokens=[9, 10, 11],
        student_action_text="ignored",
        action_tool_names=["search_corpus"],
        student_prompt_token_ids=[1, 2, 3],
    )
    datums = build_sampled_opd_datums(
        [point],
        lambda_opd=0.01,
        encode_fn=_encode,
        policy_version="v1",
        component_id="auto_populate_first_search",
        gate_beta=5.0,
    )
    assert len(datums) == 1
    d = datums[0]
    assert d.prompt_token_ids == [1, 2, 3]
    assert d.target_tokens[-3:] == [9, 10, 11]
    assert d.weights[:3] == [0.0, 0.0, 0.0]
    assert d.weights[-3:] == [1.0, 1.0, 1.0]
    assert d.n_supervised_tokens == 3
    assert d.teacher_prompt_token_ids
    assert d.metadata["lambda_opd"] == 0.01
    assert abs(supervised_weight_sum(datums) - 3.0) < 1e-9


def test_projected_seed_datums_use_target_text_and_binary_weights():
    from scape.training.tinker_opd_datum import build_projected_seed_datums

    steps = [_step("PREFIX", "aa"), _step("PREFIX", "bbb")]
    for s in steps:
        s.metadata["prompt_full"] = "FULL"
    datums = build_projected_seed_datums(
        steps, lambda_opd=0.01, encode_fn=_encode, policy_version="v1", gate_beta=5.0
    )
    assert len(datums) == 2
    assert datums[0].target_tokens[-2:] == [1, 1]
    assert datums[1].n_supervised_tokens == 3
    assert all(w in {0.0, 1.0} for d in datums for w in d.weights)
    assert datums[0].metadata["projector_used"] is True
    assert datums[0].metadata["sampled_action"] is False
    assert datums[0].teacher_prompt_token_ids == _encode("FULL")
    assert abs(supervised_weight_sum(datums) - 5.0) < 1e-9
