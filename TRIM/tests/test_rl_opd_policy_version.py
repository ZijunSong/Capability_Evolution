from __future__ import annotations

import pytest

from trim.training.rl_opd_policy_version import (
    PolicyVersionMismatch,
    assert_policy_versions_match,
)


def test_sync_versions_match():
    assert_policy_versions_match(
        rollout_policy="v10",
        train_policy="v10",
        harness_teacher_policy="v10",
    )


def test_teacher_mismatch_hard_fail():
    with pytest.raises(PolicyVersionMismatch):
        assert_policy_versions_match(
            rollout_policy="v10",
            train_policy="v10",
            harness_teacher_policy="v11",
        )


def test_train_mismatch_hard_fail():
    with pytest.raises(PolicyVersionMismatch):
        assert_policy_versions_match(
            rollout_policy="v10",
            train_policy="v11",
            harness_teacher_policy="v10",
        )
