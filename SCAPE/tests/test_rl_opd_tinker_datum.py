from __future__ import annotations

import pytest

from scape.training.opd_dataset import ProjectedTrainingStep
from scape.training.tinker_opd_datum import (
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
