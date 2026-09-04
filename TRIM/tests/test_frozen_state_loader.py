from pathlib import Path

from trim.training.frozen_state_loader import (
    groups_from_frozen_points,
    load_train_states,
    working_memory_from_state,
)
from trim.training.opd_dataset import project_and_materialize
from trim.training.opd_projection import ProjectionKind, StudentActionSpaceProjector
from trim.training.sentence_compress_teacher import teacher_events_from_point


def test_working_memory_from_event_row():
    wm = working_memory_from_state(
        {
            "query_id": "9",
            "query": "When did the author lecture?",
            "event_active": True,
            "payload": {
                "search_result_doc_ids": ["d1", "d2"],
                "observation": ("Noisy filler. " * 40) + "The author lectured from 2018.",
            },
        }
    )
    assert wm["documents"]
    assert wm["documents"][0]["id"] == "d1"


def test_load_train_states_and_project(tmp_path: Path):
    path = tmp_path / "TRAIN_STATES_5K.jsonl"
    path.write_text(
        '{"query_id":"9","query":"When did the author lecture?","event_active":true,'
        '"documents":[{"id":"d1","text":"' + ("Long noisy passage. " * 30) + 'The author lectured from 2018."}]}\n',
        encoding="utf-8",
    )
    points, meta = load_train_states(path, component_id="sentence_compress")
    assert meta["found"]
    assert meta["n_states"] == 1
    groups = groups_from_frozen_points(points)
    assert groups[0].query_id == "9"
    events = teacher_events_from_point(points[0])
    projection, steps = project_and_materialize(
        student_snapshot=points[0].pre_action_snapshot,
        teacher_events=events,
        student_mask=points[0].pre_action_snapshot.harness_mask,
        component_id="sentence_compress",
        projector=StudentActionSpaceProjector(),
    )
    assert projection.kind == ProjectionKind.DIRECT
    assert steps[0].target_action["name"] == "curate"
    assert "compressed_teacher_view" not in steps[0].prompt_reduced
