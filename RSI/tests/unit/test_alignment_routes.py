"""Alignment acceptance tests:

- test_future_observation_rejected
- test_reasoning_not_misclassified_as_leak
- test_native_anchor_without_explicit_map
"""

from __future__ import annotations

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.events import CostRecord, Event, EventKind, Visibility
from rsi.contracts.observation import Observation, ObservationProvenance, ToolSpec
from rsi.contracts.task import PublicTask
from rsi.opd.alignment import AlignmentContext, align_decision
from rsi.opd.realizability import build_realizability_report

CATALOG = (
    ToolSpec(name="read_file", description="read", parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    ToolSpec(name="write_file", description="write", parameters_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
    ToolSpec(name="get", description="kv get", parameters_schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}),
)


def _event(order: int, action, obs_content: str | None, *, visibility=Visibility.TEACHER_PRIVATE, source_ref: str = "ref"):
    obs = None
    if obs_content is not None:
        obs = Observation(
            content=obs_content,
            provenance=ObservationProvenance(source_tool="tool", source_ref=source_ref),
        )
    return Event(
        event_id=f"e{order}",
        order=order,
        component_id="teacher",
        event_kind=EventKind.TOOL_CALL if action is not None else EventKind.COMPONENT_HOOK,
        pre_state_digest="pre",
        post_state_digest="post",
        action=action,
        observation=obs,
        visibility=visibility,
    )


def test_future_observation_rejected():
    """Teacher read an unknown id; an action depending on it is rejected or
    converted to acquire-first."""
    # Teacher-private read produces id "id:xyz" that the Student has never seen.
    teacher_events = (
        _event(0, ToolCall(name="get", arguments={"key": "id:xyz"}), "secret-value", source_ref="id:xyz"),
        _event(1, ToolCall(name="write_file", arguments={"path": "out.txt", "content": "id:xyz"}), "wrote"),
    )
    student_events = ()  # student has seen nothing yet
    ctx = AlignmentContext(catalog=CATALOG, state_view={})
    result = align_decision(
        teacher_events=teacher_events,
        student_events=student_events,
        student_pre_state_digest="s0",
        ctx=ctx,
    )
    if result.route == "native_anchor":
        # must not supervise the id-dependent write
        assert result.target_action is None or result.target_action.name == "get"
        if result.target_action is not None and result.target_action.name == "get":
            # acquire-first: the get itself is the legal supervised action
            assert result.target_action.arguments.get("key") == "id:xyz"
    else:
        assert result.route == "skip"
        assert result.target_action is None

    # Direct report-level check: the write target is not realizable now.
    report = build_realizability_report(
        ToolCall(name="write_file", arguments={"path": "out.txt", "content": "id:xyz"}),
        catalog=CATALOG,
        state_view={},
        student_fact_refs=(),
        teacher_private_fact_refs=("id:xyz",),
    )
    assert not report.ok
    assert any("teacher-private" in r or "acquire" in r for r in report.reasons)


def test_reasoning_not_misclassified_as_leak():
    """Teacher's better reasoning over public inputs stays supervisable."""
    # Student has seen a.txt/b.txt; Teacher (private branch) computes the sum
    # and proposes a native write.  The numbers come from public history, the
    # *computation* is Teacher reasoning — NOT leakage.
    student_events = (
        _event(0, ToolCall(name="read_file", arguments={"path": "a.txt"}), "67", visibility=Visibility.STUDENT, source_ref="a.txt"),
        _event(1, ToolCall(name="read_file", arguments={"path": "b.txt"}), "54", visibility=Visibility.STUDENT, source_ref="b.txt"),
    )
    teacher_events = (
        _event(0, ToolCall(name="write_file", arguments={"path": "out.txt", "content": "121"}), "wrote"),
    )
    ctx = AlignmentContext(catalog=CATALOG, state_view={})
    result = align_decision(
        teacher_events=teacher_events,
        student_events=student_events,
        student_pre_state_digest="s1",
        ctx=ctx,
    )
    assert result.route == "native_anchor"
    assert result.target_action == ToolCall(name="write_file", arguments={"path": "out.txt", "content": "121"})
    assert result.legality is not None and result.legality.ok
    assert result.provenance is not None and result.provenance.verdict == "clean"


def test_native_anchor_without_explicit_map():
    """explicit_map disabled: internal events + native tool actions still
    produce valid sourced OPD supervision via skip-to-anchor."""
    student_events = (
        _event(0, ToolCall(name="read_file", arguments={"path": "a.txt"}), "1", visibility=Visibility.STUDENT, source_ref="a.txt"),
    )
    teacher_events = (
        # internal planning event (no action) — must be skipped
        _event(0, None, "planning: step 1 check"),
        _event(1, None, "planning: step 2 check"),
        _event(2, ToolCall(name="write_file", arguments={"path": "out.txt", "content": "1"}), "wrote"),
    )
    ctx = AlignmentContext(
        catalog=CATALOG,
        state_view={},
        routes_enabled=("native_anchor",),  # no explicit_map
    )
    result = align_decision(
        teacher_events=teacher_events,
        student_events=student_events,
        student_pre_state_digest="s2",
        ctx=ctx,
    )
    assert result.route == "native_anchor"
    assert result.target_action is not None
    assert result.source_event_ids  # provenance recorded
    assert result.legality.ok
