from __future__ import annotations

from scape.adapters.components import minus_mask
from scape.state.snapshot import capture_snapshot
from scape.training.opd_dataset import materialize, project_and_materialize
from scape.training.opd_events import model_action, tool_observation
from scape.training.opd_projection import ProjectionKind, StudentActionSpaceProjector
from scape.training.rl_opd_types import StudentDecisionPoint
from scape.training.tinker_opd_datum import build_tinker_opd_datums
from scape.training.tinker_rl_opd_trainer import project_on_policy_decisions


SECRET = "VERIFY_RESULT_SECRET"


def _verify_snap(**wm):
    return capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=minus_mask("verify_tool"),
        working_memory={
            "documents": [{"id": "d1"}, {"id": "d2"}],
            "curated_ids": [],
            "accessible_doc_ids": ["d1"],
            "reviewed_ids": [],
            "full_text_ids": [],
            "full_text_min_chars": 10_000,
            **wm,
        },
        metadata={"component_id": "verify_tool"},
    )


def _verify_events():
    return [
        model_action("verify", {"doc_id": "d1"}, component_id="verify_tool"),
        tool_observation(
            component_id="verify_tool",
            observation={"verdict": SECRET, "raw": SECRET},
            visible_to_student=False,
        ),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="verify_tool"),
    ]


def test_verify_secret_not_in_student_or_opd_prefix():
    snap = _verify_snap()
    projector = StudentActionSpaceProjector()
    result = projector.project(
        teacher_trace=_verify_events(),
        student_snapshot=snap,
        student_mask=snap.harness_mask,
    )
    assert result.kind == ProjectionKind.DIRECT
    steps = materialize(result, snap, component_id="verify_tool")
    assert len(steps) == 1
    blobs = [steps[0].prompt_reduced, steps[0].target_text]
    assert all(SECRET not in text for text in blobs)
    datums = build_tinker_opd_datums(steps, lambda_opd=0.1, policy_version="v0")
    assert all(SECRET not in d.model_input for d in datums)
    assert all(SECRET not in d.target_action.get("name", "") for d in datums)
    assert datums[0].teacher_prompt_token_ids


def test_aligned_curate_prefix_has_no_teacher_verify():
    snap = _verify_snap()
    _, steps = project_and_materialize(
        student_snapshot=snap,
        teacher_events=_verify_events(),
        student_mask=snap.harness_mask,
        component_id="verify_tool",
    )
    assert steps[0].target_action["name"] == "curate"
    assert SECRET not in steps[0].prompt_reduced
    assert "verdict" not in steps[0].prompt_reduced
    assert steps[0].metadata.get("prompt_full")
    assert steps[0].metadata["prompt_full"] != steps[0].prompt_reduced


def test_on_policy_projection_uses_student_snapshot_only():
    snap = _verify_snap()
    point = StudentDecisionPoint(
        episode_id="e",
        query_id="q",
        rollout_idx=0,
        turn_id=0,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="student prefix only",
        student_action_tokens=[],
        student_action_text="search_corpus",
        action_tool_names=["search_corpus"],
        reward=0.0,
        structurally_valid=True,
    )
    steps, _audit, _ = project_on_policy_decisions(
        [point],
        teacher_event_fn=lambda _p: _verify_events(),
        component_id="verify_tool",
    )
    for step in steps:
        assert SECRET not in step.prompt_reduced
        assert step.metadata.get("source_policy_version") == "v0"
        assert step.student_snapshot["query_id"] == "q"
