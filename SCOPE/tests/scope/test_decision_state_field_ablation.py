"""A7 field ablation tests."""

from __future__ import annotations

from experiments.ablations.builders.build_field_ablation import analyze_field_ablation


def test_conflicting_collision_marked_unidentifiable():
    rows = [
        {"label": "KEEP_EVIDENCE", "state": {"candidate_text": "same", "candidate_id": "1", "query": "q"}},
        {"label": "SKIP_DUPLICATE", "state": {"candidate_text": "same", "candidate_id": "2", "query": "q"}},
    ]
    # Dropping candidate_id makes them collide with conflicting labels
    report = analyze_field_ablation(rows, variant="a7_dup_no_candidate_id")
    assert report["conflicting_label_collision"] >= 1
    assert report["unidentifiable"] is True
