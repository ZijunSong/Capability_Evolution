from __future__ import annotations

from scape.adapters.components import coalition_minus_mask, full_mask
from scape.collection.same_state import audit_same_state, collect_same_state_dataset


def test_coalition_same_state_collection_uses_joint_student_mask():
    coalition = ["sentence_compress", "importance_tagging"]
    rows = collect_same_state_dataset(
        n_states=4,
        component_ids=coalition,
        seed=0,
    )
    audit = audit_same_state(rows)
    assert audit["pass"] is True
    assert audit["coalition_rows"] == 4
    assert audit["student_mask_recorded_rate"] == 1.0

    expected_mask = coalition_minus_mask(coalition)
    for row in rows:
        assert row["component_ids"] == coalition
        assert row["student_mask"] == expected_mask
        assert row["full_mask"] == full_mask()
        assert row["views_differ_by_harness_only"] is True
        assert "sentence_compress" in row["prompt_reduced"]
        assert "importance_tagging" in row["prompt_reduced"]


def test_single_component_backward_compatible_schema():
    rows = collect_same_state_dataset(n_states=2, component_id="evidence_graph", seed=1)
    row = rows[0]
    assert row["component_id"] == "evidence_graph"
    assert row["component_ids"] == ["evidence_graph"]
    assert row["student_mask"] == coalition_minus_mask(["evidence_graph"])
