from trim.adapters.components import minus_mask
from trim.state.snapshot import capture_snapshot
from trim.training.adaptive_rerank_teacher import RERANK_INSTRUCTION_KEY, teacher_events_from_wm
from trim.training.opd_dataset import project_and_materialize, prompt_has_teacher_leak
from trim.training.opd_projection import ProjectionKind, StudentActionSpaceProjector


def test_adaptive_rerank_projects_query_only_search_without_leak():
    wm = {
        "query": "Which source is direct evidence?",
        RERANK_INSTRUCTION_KEY: "prefer direct evidence",
        "documents": [{"id": "d1", "text": "evidence"}],
        "curated_ids": [],
    }
    events = teacher_events_from_wm(wm)
    assert events[0].visible_to_student is False
    assert events[0].observation[RERANK_INSTRUCTION_KEY] == "prefer direct evidence"
    assert events[1].action_name == "search_corpus"
    assert events[1].arguments == {"query": wm["query"]}
    snap = capture_snapshot(query_id="q1", step=0, harness_mask=minus_mask("adaptive_rerank_instruction"), working_memory=wm)
    projection, steps = project_and_materialize(
        student_snapshot=snap,
        teacher_events=events,
        student_mask=snap.harness_mask,
        component_id="adaptive_rerank_instruction",
        projector=StudentActionSpaceProjector(),
    )
    assert projection.kind == ProjectionKind.DIRECT
    assert steps[0].target_action["name"] == "search_corpus"
    assert not any(prompt_has_teacher_leak(s.prompt_reduced) for s in steps)
    assert not any("prefer direct evidence" in s.prompt_reduced for s in steps)
