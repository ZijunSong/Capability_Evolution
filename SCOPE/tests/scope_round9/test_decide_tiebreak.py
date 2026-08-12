"""Stable tie-break for rollback operation decisions."""

from harness.capability.rollback_operation import RollbackOperation
from training.scope.decide_rollback_operation import decide_rollback_operation


def test_exact_tie_prefers_continue():
    d = decide_rollback_operation(
        score_continue=-1.0,
        score_replan=-1.0,
        score_rollback=-1.0,
        threshold=0.0,
    )
    assert d.predicted_operation == RollbackOperation.CONTINUE


def test_near_tie_within_eps_prefers_continue():
    d = decide_rollback_operation(
        score_continue=-6.90625,
        score_replan=-13.0,
        score_rollback=-6.90624,
        threshold=0.0,
        tie_eps=1e-4,
    )
    assert d.predicted_operation == RollbackOperation.CONTINUE


def test_near_tie_requires_explicit_eps():
    d = decide_rollback_operation(
        score_continue=-11.129583299160004,
        score_replan=-16.26518726348877,
        score_rollback=-11.086169918378195,
        threshold=0.0,
        tie_eps=1e-1,
    )
    assert d.predicted_operation == RollbackOperation.CONTINUE


def test_clear_rollback_wins():
    d = decide_rollback_operation(
        score_continue=-3.0,
        score_replan=-4.0,
        score_rollback=-1.0,
        threshold=0.0,
        candidate_checkpoint_id="ck_a",
    )
    assert d.predicted_operation == RollbackOperation.ROLLBACK_TO
    assert d.checkpoint_id == "ck_a"


def test_disable_replan_ignores_high_replan_score():
    d = decide_rollback_operation(
        score_continue=-3.0,
        score_replan=10.0,
        score_rollback=-2.5,
        threshold=0.0,
        disable_replan=True,
    )
    assert d.predicted_operation == RollbackOperation.ROLLBACK_TO
