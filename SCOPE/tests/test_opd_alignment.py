"""Tests for OPD token alignment and loss."""

from __future__ import annotations

import math

from training.opd.loss import compute_opd_loss, compute_sampled_nll_loss
from training.opd.replay_buffer import OPDReplayBuffer
from training.opd._policy_backend import OPDTransition
from training.opd.token_alignment import align_action_tokens
from training.opd.trainer import OPDTrainer


def test_token_alignment():
    t = OPDTransition(
        episode_id="e",
        query_id="q",
        turn_id=0,
        student_input_ids=[1, 2, 3, 10, 11],
        action_ids=[10, 11],
        action_mask=[True, True],
        teacher_input_ids=[1, 2, 3, 4, 10, 11],
        privileged_module_id="verification",
    )
    aligned = align_action_tokens(t)
    assert aligned.action_start_student == 3
    assert aligned.action_start_teacher == 4


def test_sampled_nll_loss_finite():
    loss = compute_sampled_nll_loss([-0.5, -0.3], [1.0, 0.5])
    assert math.isfinite(loss)


def test_opd_trainer_smoke():
    trainer = OPDTrainer()
    trainer.add_transitions([
        OPDTransition(
            episode_id="e",
            query_id="q",
            turn_id=0,
            student_input_ids=[1, 2, 3],
            action_ids=[4, 5],
            action_mask=[True, True],
            teacher_input_ids=[1, 2, 3, 4, 5],
            privileged_module_id="verification",
            success=True,
        )
    ])
    metrics = trainer.train_epoch(batch_size=1)
    assert metrics["batch_size"] > 0
    assert math.isfinite(metrics["opd_loss"])
