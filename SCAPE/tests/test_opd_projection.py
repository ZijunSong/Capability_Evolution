from __future__ import annotations

from scape.adapters.components import all_component_ids, minus_mask
from scape.state.snapshot import capture_snapshot
from scape.training.opd_dataset import materialize
from scape.training.opd_events import harness_mutation, model_action, obs_transform, tool_observation
from scape.training.opd_projection import (
    ProjectionKind,
    StudentActionSpaceProjector,
)


def _snap(component_id: str, **wm_and_meta):
    meta = dict(wm_and_meta.pop("metadata", {}) or {})
    meta.setdefault("component_id", component_id)
    return capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=minus_mask(component_id),
        working_memory={
            "documents": [{"id": "d1"}, {"id": "d2"}, {"id": "d31"}],
            "curated_ids": ["d1"],
            "accessible_doc_ids": ["d1", "d2", "d31", "d9"],
            **wm_and_meta,
        },
        metadata=meta,
    )


def _project(component_id: str, events, **wm_and_meta):
    snap = _snap(component_id, **wm_and_meta)
    projector = StudentActionSpaceProjector()
    return projector.project(
        teacher_trace=events,
        student_snapshot=snap,
        student_mask=snap.harness_mask,
    ), snap


def test_all_ten_components_have_handlers():
    projector = StudentActionSpaceProjector()
    for cid in all_component_ids():
        assert cid in projector.handlers
        snap = _snap(cid)
        result = projector.project(
            teacher_trace=[obs_transform(cid, turn_id=0)],
            student_snapshot=snap,
            student_mask=snap.harness_mask,
        )
        assert result.kind in {
            ProjectionKind.DIRECT,
            ProjectionKind.MACRO,
            ProjectionKind.SKIP,
            ProjectionKind.REJECT,
        }


def test_direct_legal_search():
    events = [model_action("search_corpus", {"query": "q"}, component_id="evidence_graph")]
    result, _snap = _project("evidence_graph", events)
    assert result.kind == ProjectionKind.DIRECT
    assert result.actions[0].name == "search_corpus"


def test_sentence_compress_skip_to_curate():
    events = [
        obs_transform("sentence_compress"),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="sentence_compress"),
    ]
    result, _snap = _project("sentence_compress", events)
    assert result.kind == ProjectionKind.DIRECT
    assert "xfm" in result.skipped_event_ids[0] or result.skipped_event_ids
    assert result.actions[0].name == "curate"


def test_auto_populate_compiles_explicit_curate():
    events = [
        harness_mutation(
            "auto_populate_first_search",
            {"before_curated": ["d1"], "after_curated": ["d1", "d2"]},
        )
    ]
    result, _snap = _project("auto_populate_first_search", events)
    assert result.kind in {ProjectionKind.DIRECT, ProjectionKind.MACRO}
    action = result.actions[0]
    assert action.name == "curate"
    assert action.arguments["add_ids"] == ["d2"]
    assert "importance" not in action.arguments


def test_subtractive_curated_delta():
    events = [
        harness_mutation(
            "subtractive_curation",
            {"before_curated": ["d1", "d2"], "after_curated": ["d1"]},
        )
    ]
    result, _snap = _project(
        "subtractive_curation",
        events,
        curated_ids=["d1", "d2"],
        accessible_doc_ids=["d1", "d2"],
    )
    assert result.actions[0].name == "curate"
    assert result.actions[0].arguments["remove_ids"] == ["d2"]


def test_importance_delayed_eviction():
    events = [
        harness_mutation(
            "importance_tagging",
            {
                "importance": {"d1": "high", "d9": "low"},
                "before_curated": ["d1", "d9"],
                "after_curated": ["d1", "d9"],
            },
            turn_id=0,
        ),
        harness_mutation(
            "importance_tagging",
            {"before_curated": ["d1", "d9"], "after_curated": ["d1", "d31"]},
            turn_id=5,
        ),
    ]
    result, snap = _project(
        "importance_tagging",
        events,
        curated_ids=["d1", "d9"],
        accessible_doc_ids=["d1", "d9", "d31"],
    )
    assert result.kind in {ProjectionKind.DIRECT, ProjectionKind.MACRO}
    action = result.actions[0]
    assert action.name == "curate"
    assert action.arguments["remove_ids"] == ["d9"]
    assert action.arguments["add_ids"] == ["d31"]
    assert "importance" not in action.arguments
    steps = materialize(result, snap, component_id="importance_tagging")
    assert len(steps) == 1
    assert "importance" not in steps[0].target_text


def test_verify_macro_via_review_docs():
    events = [
        model_action("verify", {"doc_id": "d1"}, component_id="verify_tool"),
        tool_observation(component_id="verify_tool", observation={"verdict": "YES"}, visible_to_student=False),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="verify_tool"),
    ]
    result, snap = _project(
        "verify_tool",
        events,
        documents=[{"id": "d1"}, {"id": "d2"}],
        curated_ids=[],
        accessible_doc_ids=["d1"],
        reviewed_ids=[],
        full_text_ids=[],
        full_text_min_chars=10_000,
    )
    assert result.kind == ProjectionKind.MACRO
    assert [a.name for a in result.actions] == ["review_docs", "curate"]
    steps = materialize(result, snap, component_id="verify_tool")
    assert len(steps) == 2
    assert steps[0].target_action["name"] == "review_docs"
    assert steps[1].target_action["name"] == "curate"
    assert steps[0].prompt_reduced != steps[1].prompt_reduced
    assert "verify" not in steps[0].target_text
    assert "verdict" not in steps[1].prompt_reduced


def test_verify_oracle_reject():
    events = [
        model_action("verify", {"doc_id": "d1"}, component_id="verify_tool"),
        tool_observation(component_id="verify_tool", observation={"verdict": "YES"}, visible_to_student=False),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="verify_tool"),
    ]
    result, snap = _project(
        "verify_tool",
        events,
        documents=[{"id": "d2"}],
        curated_ids=[],
        accessible_doc_ids=["d2"],
    )
    assert result.kind == ProjectionKind.REJECT
    assert result.reject_reason == "TEACHER_ONLY_INFORMATION"
    assert materialize(result, snap, component_id="verify_tool") == []


def test_content_dedup_superset_still_direct():
    events = [
        obs_transform("content_dedup"),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="content_dedup"),
    ]
    result, _snap = _project(
        "content_dedup",
        events,
        documents=[{"id": "d1"}, {"id": "d2"}, {"id": "d3_dup"}, {"id": "d4"}, {"id": "d5_dup"}],
        accessible_doc_ids=["d1", "d2", "d3_dup", "d4", "d5_dup"],
    )
    assert result.kind == ProjectionKind.DIRECT
    assert result.actions[0].name == "curate"


def test_adaptive_rerank_same_tool_different_transition_not_direct():
    events = [
        model_action(
            "search_corpus",
            {"query": "q"},
            component_id="adaptive_rerank_instruction",
            state_delta={"result_ids": ["d9"]},
        )
    ]
    result, _snap = _project(
        "adaptive_rerank_instruction",
        events,
        accessible_doc_ids=["d1"],
        metadata={"student_search_results": ["d1"], "teacher_needed_doc_ids": ["d9"]},
    )
    assert result.kind == ProjectionKind.REJECT
    assert result.reject_reason == "TRANSITION_NOT_REPRODUCIBLE"
    assert result.actions == []


def test_verify_never_emitted_when_disabled():
    events = [
        model_action("verify", {"doc_id": "d1"}, component_id="verify_tool"),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, component_id="verify_tool"),
    ]
    result, snap = _project(
        "verify_tool",
        events,
        accessible_doc_ids=["d1"],
        reviewed_ids=["d1"],
        full_text_ids=["d1"],
    )
    assert all(action.name != "verify" for action in result.actions)
    for step in materialize(result, snap, component_id="verify_tool"):
        assert "to=verify" not in step.target_text
        assert "verify" not in snap.harness_mask or snap.harness_mask["verify_tool"] is False
