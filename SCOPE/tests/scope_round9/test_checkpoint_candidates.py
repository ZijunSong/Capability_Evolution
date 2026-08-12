"""Unit tests for checkpoint candidate ordering."""

from training.scope.checkpoint_candidates import (
    assign_local_checkpoint_ids,
    global_to_local_id,
    order_checkpoint_candidates,
)


def test_ordering_is_deterministic():
    cands = [
        {"checkpoint_id": "b", "turn_id": 1, "n_curated": 2},
        {"checkpoint_id": "a", "turn_id": 2, "n_curated": 1},
        {"checkpoint_id": "c", "turn_id": 2, "n_curated": 1},
    ]
    ordered = order_checkpoint_candidates(cands)
    assert [c["checkpoint_id"] for c in ordered] == ["a", "c", "b"]


def test_local_ids_assigned():
    cands = [{"checkpoint_id": "x", "turn_id": 0}, {"checkpoint_id": "y", "turn_id": 1}]
    enriched, mapping = assign_local_checkpoint_ids(cands)
    assert enriched[0]["local_checkpoint_id"] == "C0"
    assert mapping["C0"] == "y"
    assert global_to_local_id("y", mapping) == "C0"
