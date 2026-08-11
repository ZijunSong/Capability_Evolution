from __future__ import annotations

from scape.adapters.components import minus_mask
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import capture_snapshot
from scape.training.teacher import FullViewTeacher


def _snap():
    return capture_snapshot(
        query_id="q-dv",
        step=1,
        harness_mask=minus_mask("importance_tagging"),
        working_memory={
            "documents": [{"id": "d1", "text": "abcdef" * 20}],
            "curated_docs": [{"id": "d1", "text": "abcdef" * 20}],
            "curated_importance": {"d1": "high"},
            "evidence_graph": {"nodes": ["d1"], "edges": []},
        },
    )


def test_dual_view_same_snapshot():
    snap = _snap()
    rend = DualViewRenderer()
    dual = rend.render_pair(snap, component_id="importance_tagging")
    rend.assert_same_snapshot(dual, snap)
    assert dual.snapshot_hash == snap.content_hash()
    assert dual.student_view["query_id"] == dual.full_view["query_id"] == snap.query_id


def test_full_teacher_does_not_step_environment():
    snap = _snap()
    teacher = FullViewTeacher()
    before = teacher.renderer.environment_steps
    dual = teacher.dual_view(snap, component_id="importance_tagging")
    _ = teacher.score(dual.full_view)
    teacher.assert_no_env_step()
    assert teacher.renderer.environment_steps == before
