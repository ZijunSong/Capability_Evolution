from scape.adapters.components import minus_mask
from scape.state.snapshot import capture_snapshot
from scape.training.opd_dataset import project_and_materialize, prompt_has_teacher_leak
from scape.training.opd_projection import ProjectionKind, StudentActionSpaceProjector
from scape.training.sentence_compress_teacher import (
    COMPRESSED_VIEW_KEY,
    compress_text,
    is_compression_active_state,
    teacher_events_from_wm,
)


def test_compress_keeps_query_overlap():
    text = "Unrelated weather. The author lectured at a private university from 2018 until his death. Sports scores."
    out = compress_text("author lectured university 2018", text, max_sents=1)
    assert "university" in out.lower()


def test_teacher_projects_to_student_curate_without_leak():
    wm = {
        "query": "When did the author lecture?",
        "documents": [
            {"id": "ev1", "text": ("Noisy filler. " * 30) + "The author lectured from 2018 until his death."},
            {"id": "noise", "text": "Sports scores and travel delays. " * 20},
        ],
        "curated_ids": [],
    }
    assert is_compression_active_state(wm)
    events = teacher_events_from_wm(wm)
    assert events[0].kind.value == "obs_transform"
    assert events[0].visible_to_student is False
    assert COMPRESSED_VIEW_KEY in (events[0].observation or {})
    snap = capture_snapshot(
        query_id="1",
        step=0,
        harness_mask=minus_mask("sentence_compress"),
        working_memory=wm,
    )
    projection, steps = project_and_materialize(
        student_snapshot=snap,
        teacher_events=events,
        student_mask=snap.harness_mask,
        component_id="sentence_compress",
        projector=StudentActionSpaceProjector(),
    )
    assert projection.kind == ProjectionKind.DIRECT
    assert steps
    assert steps[0].target_action["name"] == "curate"
    assert not any(prompt_has_teacher_leak(s.prompt_reduced) for s in steps)
    assert not any(COMPRESSED_VIEW_KEY in s.prompt_reduced for s in steps)
