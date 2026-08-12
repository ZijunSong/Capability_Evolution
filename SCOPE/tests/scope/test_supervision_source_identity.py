"""A1 supervision source identity tests."""

from __future__ import annotations

from experiments.ablations.builders.build_supervision_source import (
    build_cross_state_matched,
    build_from_paths,
    build_same_state_on_policy,
    state_hash,
)


def test_same_state_hash_identity():
    live = [{"query_id": "q0", "turn": 0, "candidate_text": "a", "candidate_id": "c0"}]
    labels = [{"label": "KEEP_EVIDENCE"}]
    rows = build_same_state_on_policy(live, shadow_labels=labels)
    assert rows[0]["live_state_hash"] == rows[0]["source_state_hash"]
    assert rows[0]["live_state_hash"] == state_hash(live[0])


def test_cross_state_hash_differs():
    live = [{"query_id": "q0", "turn": 0, "candidate_text": "a", "candidate_id": "c0"}]
    pool = [{"query_id": "q0", "turn": 1, "candidate_text": "b", "candidate_id": "c1"}]
    labels = [{"label": "SKIP_DUPLICATE"}]
    rows = build_cross_state_matched(live, pool, shadow_labels=labels)
    assert rows[0]["live_state_hash"] != rows[0]["source_state_hash"]


def test_all_variants_at_least_16():
    for v in [
        "a1_same_state_on_policy",
        "a1_trajectory_teacher",
        "a1_cross_state_matched",
        "a1_static_offline",
    ]:
        result = build_from_paths(v, n_target=16)
        assert result["report"]["n_samples"] >= 16
