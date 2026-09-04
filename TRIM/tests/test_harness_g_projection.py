"""Action projection for Harness-G advanced components (OPD skip-to-anchor)."""

from __future__ import annotations

from trim.adapters.components import zero_mask
from trim.state.snapshot import capture_snapshot
from trim.training.opd_events import model_action, obs_transform
from trim.training.opd_projection import ProjectionKind, StudentActionSpaceProjector
from trim.training.harness_g_projection import project_harness_g_events
from trim.training.harness_g_teacher import answer_with_events_from_wm, snc_frontier_events_from_wm


def _g_snap(**wm):
    base = {
        "visible_sids": ["d1:s0", "d1:s1"],
        "selected_sids": [],
        "frontier_eids": ["e:alice_smith"],
        "accessible_doc_ids": ["d1:s0", "d1:s1", "e:alice_smith"],
        "sentences": {
            "d1:s0": {"sid": "d1:s0", "text": "Alice Smith visited Paris.", "doc_id": "d1"},
            "d1:s1": {"sid": "d1:s1", "text": "Bob Jones lived nearby.", "doc_id": "d1"},
        },
        "entities": {
            "e:alice_smith": {"eid": "e:alice_smith", "surface": "Alice Smith", "sids": ["d1:s0"]},
            "e:hidden_bridge": {"eid": "e:hidden_bridge", "surface": "Carol Adams", "sids": ["d2:s0"]},
        },
        "action_map": {
            "A0": {"type": "SELECT", "sid": "d1:s0", "name": "select", "snc_preview": 0.9},
            "A1": {"type": "LOOKUP", "eid": "e:alice_smith", "name": "lookup", "snc_preview": 0.1},
        },
        "initialized": True,
    }
    base.update(wm)
    return capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=zero_mask("Harness-G"),
        working_memory=base,
        metadata={"harness": "Harness-G", "component_id": "answer_with"},
    )


def _project(events, snap=None, component_id="answer_with"):
    snap = snap or _g_snap()
    projector = StudentActionSpaceProjector()
    return projector.project(
        teacher_trace=events,
        student_snapshot=snap,
        student_mask=snap.harness_mask,
    ), snap


def test_g_components_registered_on_projector():
    projector = StudentActionSpaceProjector()
    for cid in zero_mask("Harness-G"):
        assert cid in projector.handlers


def test_answer_with_projects_to_select():
    events = [
        model_action(
            "answer_with",
            {"sid": "d1:s0", "sids": ["d1:s0"]},
            component_id="answer_with",
            visible_to_student=False,
            metadata={
                "teacher_only": True,
                "projectable_target": {"name": "select", "arguments": {"sid": "d1:s0"}},
            },
        )
    ]
    result, _snap = _project(events)
    assert result.kind == ProjectionKind.DIRECT
    assert result.actions[0].name == "select"
    assert result.actions[0].arguments["sid"] == "d1:s0"


def test_answer_with_teacher_helper_projects():
    snap = _g_snap()
    events = answer_with_events_from_wm(snap.working_memory)
    result = project_harness_g_events(events, student_snapshot=snap, component_id="answer_with")
    assert result.kind == ProjectionKind.DIRECT
    assert result.actions[0].name == "select"


def test_bridge_lookup_unreachable_skipped_then_select():
    events = [
        obs_transform(
            "bridge_entities",
            observation={"bridge_eid": "e:hidden_bridge"},
            visible_to_student=False,
            metadata={"event_type": "bridge_entities_privileged_context", "harness_only": True},
        ),
        model_action(
            "lookup",
            {"eid": "e:hidden_bridge"},
            component_id="bridge_entities",
            metadata={"bridge_lookup": True},
        ),
        model_action("select", {"sid": "d1:s0"}, component_id="bridge_entities"),
    ]
    snap = _g_snap()
    result, _ = _project(events, snap=snap, component_id="bridge_entities")
    assert result.kind == ProjectionKind.DIRECT
    assert result.skipped_event_ids
    assert result.actions[0].name == "select"
    assert result.actions[0].arguments["sid"] == "d1:s0"


def test_bridge_lookup_only_unreachable_is_skip():
    events = [
        model_action("lookup", {"eid": "e:hidden_bridge"}, component_id="bridge_entities"),
    ]
    result, _ = _project(events)
    assert result.kind == ProjectionKind.SKIP
    assert not result.actions


def test_snc_obs_skip_then_align_select():
    snap = _g_snap()
    events = snc_frontier_events_from_wm(snap.working_memory)
    result = project_harness_g_events(events, student_snapshot=snap, component_id="snc_frontier")
    assert result.kind == ProjectionKind.DIRECT
    assert result.skipped_event_ids
    assert result.actions[0].name == "select"
    assert result.actions[0].arguments["sid"] == "d1:s0"


def test_direct_student_select_is_not_skipped():
    events = [model_action("select", {"sid": "d1:s0"}, component_id="entity_synonyms")]
    result, _ = _project(events)
    assert result.kind == ProjectionKind.DIRECT
    assert result.actions[0].name == "select"
    assert not result.skipped_event_ids
