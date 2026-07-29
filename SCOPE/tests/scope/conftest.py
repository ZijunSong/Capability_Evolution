"""Shared fixtures for SCOPE v3 protocol tests."""

from __future__ import annotations

import pytest

from harness.capability.state import DecisionState, VerificationRecordState


@pytest.fixture
def base_state_kwargs() -> dict:
    return dict(
        episode_id="ep1",
        task_id="t1",
        turn_id=3,
        query="Who invented X?",
        rendered_context="doc d1 mentions X inventor",
        action_history=(),
        observation_ids=("obs_1",),
        visible_document_ids=("d1", "d2"),
        pool_document_ids=("d1", "d2"),
        curated_document_ids=("d1",),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=5,
        remaining_search_calls=None,
        token_budget_used=10,
        token_budget_total=100,
        last_action_type="curate_document",
        repeated_query_score=0.0,
        wm_snapshot_hash="h",
    )


def make_state(**kwargs) -> DecisionState:
    base = dict(
        episode_id="ep1",
        task_id="t1",
        turn_id=3,
        query="Who invented X?",
        rendered_context="doc d1 mentions X inventor",
        action_history=(),
        observation_ids=("obs_1",),
        visible_document_ids=("d1", "d2"),
        pool_document_ids=("d1", "d2"),
        curated_document_ids=("d1",),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=5,
        remaining_search_calls=None,
        token_budget_used=10,
        token_budget_total=100,
        last_action_type="curate_document",
        repeated_query_score=0.0,
        wm_snapshot_hash="h",
    )
    base.update(kwargs)
    return DecisionState(**base)


def verified_stop_state() -> DecisionState:
    return make_state(
        curated_document_ids=("d1",),
        verification_records=(
            VerificationRecordState(
                turn_id=1,
                claim="X was invented by Y",
                document_ids=("d1",),
                judgments={"d1": True},
            ),
        ),
    )
