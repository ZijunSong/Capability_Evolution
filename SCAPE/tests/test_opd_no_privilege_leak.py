from __future__ import annotations

from scape.adapters.components import minus_mask
from scape.state.snapshot import capture_snapshot
from scape.training.opd_dataset import materialize, prompt_has_teacher_leak, render_student_prompt
from scape.training.opd_events import model_action, tool_observation
from scape.training.opd_projection import StudentActionSpaceProjector
from scape.training.opd_realizability import apply_student_action


def test_student_prompt_strips_teacher_only_markers():
    snap = capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=minus_mask("verify_tool"),
        working_memory={
            "documents": [{"id": "d1", "text": "raw"}],
            "curated_ids": ["d1"],
            "accessible_doc_ids": ["d1"],
            "curated_importance": {"d1": "high"},
            "teacher_verify_judgment": "YES",
        },
        metadata={"component_id": "verify_tool"},
    )
    prompt = render_student_prompt(snap, component_id="verify_tool")
    assert "teacher_verify_judgment" not in prompt
    assert "YES" not in prompt or "verify" not in prompt.lower()
    assert prompt_has_teacher_leak("teacher_verify_judgment: YES") is True
    assert prompt_has_teacher_leak(prompt) is False


def test_importance_table_not_in_reduced_prompt():
    snap = capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=minus_mask("importance_tagging"),
        working_memory={
            "documents": [{"id": "d1", "text": "raw"}],
            "curated_ids": ["d1"],
            "curated_importance": {"d1": "secret_priority"},
        },
    )
    prompt = render_student_prompt(snap, component_id="importance_tagging")
    assert "secret_priority" not in prompt
    assert "importance_table" not in prompt


def test_macro_does_not_copy_teacher_future_observation():
    snap = capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=minus_mask("verify_tool"),
        working_memory={
            "documents": [{"id": "d1"}],
            "curated_ids": [],
            "accessible_doc_ids": ["d1"],
            "reviewed_ids": [],
            "full_text_ids": [],
            "full_text_min_chars": 10_000,
        },
        metadata={"component_id": "verify_tool"},
    )
    events = [
        model_action("verify", {"doc_id": "d1"}, component_id="verify_tool"),
        tool_observation(
            component_id="verify_tool",
            observation={"teacher_only_observation": "hidden_graph", "verdict": "YES"},
            visible_to_student=False,
        ),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="verify_tool"),
    ]
    projection = StudentActionSpaceProjector().project(
        teacher_trace=events,
        student_snapshot=snap,
        student_mask=snap.harness_mask,
    )
    steps = materialize(projection, snap, component_id="verify_tool")
    assert steps
    for step in steps:
        assert "hidden_graph" not in step.prompt_reduced
        assert "teacher_only_observation" not in step.prompt_reduced
        restored = snap.__class__.from_dict(step.student_snapshot)
        restored.assert_no_future(max_known_step=restored.step)


def test_student_shadow_never_merges_teacher_state():
    snap = capture_snapshot(
        query_id="q",
        step=2,
        harness_mask=minus_mask("verify_tool"),
        working_memory={"curated_ids": ["d1"], "accessible_doc_ids": ["d1"]},
    )
    nxt = apply_student_action(snap, {"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": []}})
    nxt.assert_no_future(max_known_step=nxt.step)
    assert nxt.step == 3
    assert all(int(obs.get("step", 0)) <= nxt.step for obs in nxt.observations)
    assert snap.step == 2
