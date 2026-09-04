from __future__ import annotations

import pytest

from trim.adapters.components import minus_mask
from trim.state.snapshot import capture_snapshot, snapshot_roundtrip_ok


def test_snapshot_no_future_information():
    snap = capture_snapshot(
        query_id="q1",
        step=2,
        harness_mask=minus_mask("evidence_graph"),
        working_memory={"documents": []},
        tool_history=[{"step": 1, "action": {"name": "search"}}],
        observations=[{"step": 2, "ok": True}],
    )
    snap.assert_no_future(max_known_step=2)

    with pytest.raises(AssertionError):
        bad = capture_snapshot(
            query_id="q1",
            step=2,
            harness_mask=minus_mask("evidence_graph"),
            working_memory={},
            observations=[{"step": 3, "ok": True}],
        )
        bad.assert_no_future(max_known_step=2)


def test_snapshot_roundtrip_hash():
    snap = capture_snapshot(
        query_id="q2",
        step=1,
        harness_mask=minus_mask("content_dedup"),
        working_memory={"documents": [{"id": "d1", "text": "hello"}]},
        tool_history=[{"step": 0, "action": {"name": "search"}}],
    )
    assert snapshot_roundtrip_ok(snap)
    h1 = snap.content_hash()
    snap2 = capture_snapshot(
        query_id="q2",
        step=1,
        harness_mask=minus_mask("content_dedup"),
        working_memory={"documents": [{"id": "d1", "text": "hello"}]},
        tool_history=[{"step": 0, "action": {"name": "search"}}],
    )
    assert snap2.content_hash() == h1
