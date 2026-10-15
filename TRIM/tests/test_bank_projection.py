from trim.adapters.components import zero_mask
from trim.harness_bank.adaptation import project_ordered_events
from trim.state.snapshot import EnvironmentSnapshot
from trim.training.opd_events import EventKind, HarnessEvent
from trim.training.opd_projection import ProjectionKind


def _snap(**wm) -> EnvironmentSnapshot:
    memory = {"query": "q", "accessible_doc_ids": ["seen"]}
    memory.update(wm)
    return EnvironmentSnapshot(
        query_id="q",
        step=0,
        harness_mask=zero_mask("Harness-1"),
        working_memory=memory,
        query_text="q",
    )


def _event(name: str, arguments: dict, **meta) -> HarnessEvent:
    return HarnessEvent(
        event_id=name,
        turn_id=0,
        kind=EventKind.MODEL_ACTION,
        component_id="evo.repair",
        action_name=name,
        arguments=arguments,
        metadata=meta,
    )


def test_unseen_doc_curate_is_rejected():
    result = project_ordered_events(
        [_event("curate", {"add_ids": ["hidden"]}, information_provenance=[{"available_to_student": False}])],
        _snap(),
        zero_mask("Harness-1"),
    )
    assert result.kind == ProjectionKind.SKIP
    assert "TEACHER_ONLY" in (result.reject_reason or "")


def test_capacity_and_noop_and_illegal_tool_are_rejected():
    mask = zero_mask("Harness-1")
    capacity = project_ordered_events(
        [_event("curate", {"add_ids": ["seen"]})],
        _snap(curated_ids=["a", "b", "c", "d", "e", "f", "g"]),
        mask,
        max_curated=7,
    )
    assert capacity.kind == ProjectionKind.SKIP
    noop = project_ordered_events(
        [_event("curate", {"add_ids": ["seen"]})],
        _snap(curated_ids=["seen"], accessible_doc_ids=["seen"]),
        mask,
    )
    assert noop.reject_reason == "NO_OP_ACTION"
    illegal = project_ordered_events(
        [_event("verify", {"claim": "x"})],
        _snap(),
        mask,
    )
    assert illegal.kind == ProjectionKind.SKIP


def test_future_doc_dependency_is_rejected_and_native_search_passes():
    hidden = HarnessEvent(
        event_id="hidden",
        turn_id=0,
        kind=EventKind.HARNESS_MUTATION,
        component_id="evo.repair",
        metadata={"introduces_unseen_docs": True, "new_doc_ids": ["secret"]},
    )
    curate = _event("curate", {"add_ids": ["secret"]})
    rejected = project_ordered_events([hidden, curate], _snap(accessible_doc_ids=["secret"]), zero_mask("Harness-1"))
    assert rejected.reject_reason == "DEPENDS_ON_FUTURE_STATE"
    accepted = project_ordered_events(
        [_event("search_corpus", {"query": "visible filing"})],
        _snap(),
        zero_mask("Harness-1"),
    )
    assert accepted.kind == ProjectionKind.DIRECT
    assert accepted.actions[0].name == "search_corpus"
