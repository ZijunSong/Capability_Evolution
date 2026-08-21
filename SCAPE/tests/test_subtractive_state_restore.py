from __future__ import annotations

from scape.adapters.components import minus_mask
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot, capture_snapshot, snapshot_roundtrip_ok


def test_subtractive_state_restore_preserves_documents_and_curated_ids():
    snap = capture_snapshot(
        query_id="restore-q",
        step=2,
        harness_mask=minus_mask("subtractive_curation"),
        working_memory={
            "documents": [{"id": "d1", "text": "gold"}, {"id": "d2", "text": "noise"}],
            "curated_ids": ["d2"],
            "curated_docs": [{"id": "d2", "text": "noise"}],
        },
        tool_history=[{"step": 1, "action": {"name": "review_docs", "arguments": {"doc_ids": ["d1", "d2"]}}}],
    )
    assert snapshot_roundtrip_ok(snap)
    restored = EnvironmentSnapshot.from_dict(snap.to_dict())
    assert restored.working_memory["documents"] == snap.working_memory["documents"]
    assert restored.working_memory["curated_ids"] == ["d2"]

    dual = DualViewRenderer().render_pair(restored, component_id="subtractive_curation")
    assert dual.snapshot_hash == restored.content_hash()
    assert dual.student_view["documents"]
    assert dual.full_view["documents"]
