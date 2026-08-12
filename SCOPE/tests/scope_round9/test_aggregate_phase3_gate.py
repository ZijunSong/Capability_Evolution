"""Synthetic tests for corrected Phase 3 gate aggregation."""

from __future__ import annotations

from training.scope_round9.aggregate_phase3_gate import aggregate_events


def _event(
    *,
    shadow_op: str,
    student_op: str,
    shadow_ck: str | None = None,
    pred_ck: str | None = None,
    restore: bool | None = None,
    candidates: list[str] | None = None,
    budget_violation: bool = False,
    fallback_reason: str | None = None,
) -> dict:
    row = {
        "shadow_operation": shadow_op,
        "student_operation": student_op,
        "shadow_checkpoint_id": shadow_ck,
        "predicted_checkpoint_id": pred_ck,
    }
    if restore is not None:
        row["state_hash_restore"] = restore
    if candidates is not None:
        row["candidate_checkpoint_ids"] = candidates
    if budget_violation:
        row["budget_violation"] = True
    if fallback_reason:
        row["fallback_reason"] = fallback_reason
    return row


def test_correct_restore_rate():
    events = [
        _event(shadow_op="ROLLBACK_TO", student_op="ROLLBACK_TO", restore=True),
        _event(shadow_op="ROLLBACK_TO", student_op="ROLLBACK_TO", restore=False),
        _event(shadow_op="CONTINUE", student_op="CONTINUE", restore=True),
    ]
    m = aggregate_events(events)
    assert m["n_rollback_executed"] == 2
    assert m["state_hash_restore_rate"] == 0.5


def test_hash_mismatch_not_counted_in_numerator():
    events = [
        _event(shadow_op="ROLLBACK_TO", student_op="ROLLBACK_TO", restore=False),
    ]
    m = aggregate_events(events)
    assert m["state_hash_restore_rate"] == 0.0


def test_budget_exhaustion_only_counts_violations():
    events = [
        _event(shadow_op="CONTINUE", student_op="CONTINUE"),
        _event(shadow_op="REPLAN", student_op="REPLAN", budget_violation=True),
    ]
    m = aggregate_events(events)
    assert m["budget_violations"] == 1


def test_fallback_counted():
    events = [
        _event(shadow_op="CONTINUE", student_op="CONTINUE", fallback_reason="scorer_error"),
    ]
    m = aggregate_events(events)
    assert m["fallback_count"] == 1


def test_checkpoint_only_on_gold_rollback_with_candidates():
    events = [
        _event(
            shadow_op="ROLLBACK_TO",
            student_op="ROLLBACK_TO",
            shadow_ck="ck_a",
            pred_ck="ck_a",
            candidates=["ck_a", "ck_b"],
        ),
        _event(
            shadow_op="ROLLBACK_TO",
            student_op="ROLLBACK_TO",
            shadow_ck="ck_missing",
            pred_ck="ck_a",
            candidates=["ck_a"],
        ),
        _event(
            shadow_op="CONTINUE",
            student_op="ROLLBACK_TO",
            shadow_ck=None,
            pred_ck="ck_a",
            candidates=["ck_a"],
        ),
    ]
    m = aggregate_events(events)
    assert m["n_checkpoint_eval"] == 1
    assert m["target_checkpoint_accuracy"] == 1.0


def test_no_rollback_events_restore_denominator_zero():
    events = [
        _event(shadow_op="CONTINUE", student_op="CONTINUE"),
        _event(shadow_op="REPLAN", student_op="REPLAN"),
    ]
    m = aggregate_events(events)
    assert m["state_hash_restore_rate"] == 0.0
    assert m["n_rollback_executed"] == 0


def test_operation_balanced_accuracy_excludes_checkpoint():
    events = [
        _event(shadow_op="CONTINUE", student_op="CONTINUE"),
        _event(shadow_op="REPLAN", student_op="REPLAN"),
        _event(
            shadow_op="ROLLBACK_TO",
            student_op="ROLLBACK_TO",
            shadow_ck="ck_a",
            pred_ck="ck_b",
            candidates=["ck_a", "ck_b"],
        ),
    ]
    m = aggregate_events(events)
    assert m["operation_balanced_accuracy"] == 1.0
    assert m["target_checkpoint_accuracy"] == 0.0
