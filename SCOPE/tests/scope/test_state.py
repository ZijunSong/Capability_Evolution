"""DecisionState immutability and serialization tests."""

from __future__ import annotations

import pytest

from harness.capability.state import (
    ActionRecord,
    DecisionState,
    ObservationRecord,
    compute_wm_snapshot_hash,
)
from harness.ultra_core import WorkingMemory


def _sample_state() -> DecisionState:
    return DecisionState(
        episode_id="ep1",
        task_id="t1",
        turn_id=2,
        query="q",
        rendered_context="ctx",
        action_history=(ActionRecord(turn_id=1, action_type="search", arguments={"query": "a"}),),
        observation_ids=("obs_1",),
        visible_document_ids=("d1",),
        pool_document_ids=("d1", "d2"),
        curated_document_ids=("d1",),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=5,
        remaining_search_calls=None,
        token_budget_used=10,
        token_budget_total=100,
        last_action_type="search",
        repeated_query_score=0.1,
        wm_snapshot_hash="hash",
    )


def test_decision_state_immutable():
    s = _sample_state()
    with pytest.raises(Exception):
        s.turn_id = 99  # type: ignore[misc]


def test_serialization_round_trip():
    s = _sample_state()
    s2 = DecisionState.from_dict(s.to_dict())
    assert s2.episode_id == s.episode_id
    assert s2.observation_ids == s.observation_ids
    assert s2.state_hash() == s.state_hash()


def test_snapshot_hash_stable():
    h1 = compute_wm_snapshot_hash(
        curated_ids=["a"], pool_ids=["a", "b"], search_history=["x"], turn_number=1
    )
    h2 = compute_wm_snapshot_hash(
        curated_ids=["a"], pool_ids=["a", "b"], search_history=["x"], turn_number=1
    )
    assert h1 == h2


def test_no_invisible_docs_in_export_fields():
    s = _sample_state()
    d = s.to_dict()
    assert "doc_store" not in d
    assert set(d["visible_document_ids"]).issubset(set(d["pool_document_ids"]))


def test_observation_lineage_on_wm():
    wm = WorkingMemory("q")
    oid = wm.record_observation(
        source_type="search",
        source_document_ids=["d1"],
        text="hello",
        visible_to_student=True,
    )
    assert oid.startswith("obs_")
    assert len(wm.observation_lineage) == 1
    assert wm.observation_lineage[0]["text_hash"]
    h1 = wm.snapshot_hash()
    h2 = wm.snapshot_hash()
    assert h1 == h2


def test_observation_record_round_trip():
    r = ObservationRecord(
        observation_id="o1",
        source_type="verify",
        source_document_ids=("d1",),
        created_turn=3,
        visible_to_student=True,
        text_hash="abcd",
    )
    r2 = ObservationRecord.from_dict(r.to_dict())
    assert r2 == r
