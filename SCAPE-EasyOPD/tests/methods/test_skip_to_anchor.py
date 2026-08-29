from __future__ import annotations

from easyopd.methods.scape_component_opd.skip_to_anchor import (
    ALIGN,
    SKIP,
    project_bridge_steps,
    project_events,
)


def test_auto_side_effect_aligns_to_student_curate() -> None:
    events = [
        {
            "event_id": "auto0",
            "kind": "component_event",
            "event_type": "first_successful_search_auto_populate",
            "state_delta": {"before_curated": ["d1"], "after_curated": ["d1", "d7"]},
            "accessible_doc_ids": ["d1", "d7"],
        }
    ]
    results = project_events(events, accessible_doc_ids=["d1", "d7"], first_only=True)
    assert len(results) == 1
    assert results[0].kind == ALIGN
    assert results[0].actions[0].name == "curate"
    assert results[0].actions[0].arguments["add_ids"] == ["d7"]


def test_graph_is_epsilon_then_aligns_downstream_curate() -> None:
    events = [
        {
            "event_id": "graph0",
            "kind": "component_event",
            "event_type": "evidence_graph_privileged_context",
            "harness_only": True,
            "accessible_doc_ids": ["d1", "d2", "d3"],
        },
        {
            "event_id": "curate0",
            "kind": "model_action",
            "action_name": "curate",
            "arguments": {"add_ids": ["d3"], "remove_ids": []},
            "accessible_doc_ids": ["d1", "d2", "d3"],
        },
    ]
    results = project_events(events, accessible_doc_ids=["d1", "d2", "d3"], first_only=True)
    assert results[0].kind == ALIGN
    assert results[0].skipped_event_ids == ["graph0"]
    assert results[0].actions[0].name == "curate"
    assert results[0].actions[0].arguments["add_ids"] == ["d3"]


def test_unrealizable_downstream_curate_is_skipped_not_macro() -> None:
    events = [
        {
            "event_id": "graph0",
            "kind": "component_event",
            "event_type": "evidence_graph_privileged_context",
            "harness_only": True,
        },
        {
            "event_id": "curate0",
            "kind": "model_action",
            "action_name": "curate",
            "arguments": {"add_ids": ["secret"], "remove_ids": []},
        },
    ]
    results = project_events(events, accessible_doc_ids=["d1"])
    assert results[0].kind == SKIP
    assert results[0].actions == []


def test_verify_is_epsilon_not_a_student_label() -> None:
    events = [
        {"event_id": "v0", "kind": "model_action", "action_name": "verify", "arguments": {"doc_id": "d1"}},
        {"event_id": "o0", "kind": "tool_observation", "visible_to_student": False},
        {
            "event_id": "c0",
            "kind": "model_action",
            "action_name": "curate",
            "arguments": {"add_ids": ["d1"], "remove_ids": []},
            "accessible_doc_ids": ["d1"],
        },
    ]
    results = project_events(events, accessible_doc_ids=["d1"], first_only=True)
    assert results[0].kind == ALIGN
    assert [a.name for a in results[0].actions] == ["curate"]
    assert "v0" in results[0].skipped_event_ids


def test_projector_emits_only_align_or_skip() -> None:
    mixed = [
        {"event_id": "g", "kind": "component_event", "event_type": "evidence_graph_privileged_context", "harness_only": True},
        {"event_id": "v", "kind": "model_action", "action_name": "verify", "arguments": {"doc_id": "d1"}},
        {
            "event_id": "c",
            "kind": "model_action",
            "action_name": "curate",
            "arguments": {"add_ids": ["d1"], "remove_ids": []},
            "accessible_doc_ids": ["d1"],
        },
    ]
    for result in project_events(mixed, accessible_doc_ids=["d1"]):
        assert result.kind in {ALIGN, SKIP}
        if result.kind == ALIGN:
            assert result.actions and result.actions[0].name in {
                "fan_out_search",
                "search_corpus",
                "grep_corpus",
                "read_document",
                "review_docs",
                "curate",
                "end_search",
            }
            assert result.actions[0].name != "verify"
        else:
            assert result.actions == []


def test_bridge_steps_graph_then_curate() -> None:
    steps = [
        {
            "event": {
                "component": "evidence_graph",
                "event_type": "evidence_graph_privileged_context",
                "event_active": True,
                "harness_only": True,
                "payload": {},
            },
            "student_action": {"tool_name": "search_corpus", "parameters": {"query": "q"}},
            "pre_state": {"student_observable_env_state": {"visible_doc_ids": []}},
            "post_state": {"student_observable_env_state": {"visible_doc_ids": ["d3"]}},
        },
        {
            "event": None,
            "student_action": {"tool_name": "curate", "parameters": {"add_ids": ["d3"], "remove_ids": []}},
            "pre_state": {"student_observable_env_state": {"visible_doc_ids": ["d3"]}},
            "post_state": {"student_observable_env_state": {"visible_doc_ids": ["d3"]}},
        },
    ]
    results = project_bridge_steps(steps, component_id="evidence_graph")
    assert results[0].kind == ALIGN
    assert results[0].actions[0].name == "curate"
    assert results[0].skipped_event_ids == ["evt_0"]
