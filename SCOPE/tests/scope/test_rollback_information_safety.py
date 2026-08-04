"""Rollback DecisionState information safety."""

from __future__ import annotations

from harness.capability.state import DecisionState
from training.scope.rollback_decision_state import build_rollback_decision_state


def test_decision_state_no_gold_or_future_fields():
    base = DecisionState.from_dict(
        {
            "episode_id": "e1",
            "task_id": "q1",
            "turn_id": 1,
            "query": "test query",
            "rendered_context": "ctx",
            "action_history": [],
            "observation_ids": [],
            "visible_document_ids": [],
            "pool_document_ids": [],
            "curated_document_ids": [],
            "evidence_claims": [],
            "verification_records": [],
            "remaining_turns": 10,
            "remaining_search_calls": 5,
            "token_budget_used": 0,
            "token_budget_total": 1000,
            "last_action_type": "",
            "repeated_query_score": 0.0,
            "wm_snapshot_hash": "abc",
        }
    )
    ds = build_rollback_decision_state(
        base,
        recent_queries=["a"],
        available_checkpoints=[{"checkpoint_id": "c1", "state_hash": "h1"}],
        state_hash="h1",
    )
    assert "gold_answer" not in ds
    assert "future_trajectory" not in ds
    assert ds["capability"] == "rollback_decision"
    assert ds["available_checkpoints"][0]["checkpoint_id"] == "c1"
