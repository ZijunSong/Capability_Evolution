from __future__ import annotations

from scape.collection.same_state import audit_same_state, collect_same_state_dataset
from scape.training.hf_tool_opd import assert_loss_paths_distinct


def test_same_state_collection_contract():
    rows = collect_same_state_dataset(n_states=4, component_id="evidence_graph", seed=0)
    audit = audit_same_state(rows)
    assert audit["pass"] is True
    assert audit["legacy_scope_path_used"] is False
    assert all(r["legacy_scope_path_used"] is False for r in rows)
    assert all(r["views_differ_by_harness_only"] for r in rows)


def test_loss_paths_are_code_distinct():
    info = assert_loss_paths_distinct()
    assert info["distinct"] is True
