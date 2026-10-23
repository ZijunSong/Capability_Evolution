"""B-route: skip-to-anchor alignment over Teacher events.

Internal planning / checking / context / memory events are skipped; supervision
forms on later Student-native (basic) actions.  Skipping internal events never
licenses skipping *environment-information acquisition*: the three-layer
realizability + provenance checks still apply (see realizability.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rsi.contracts.action import Action
from rsi.contracts.events import Event, EventKind
from rsi.contracts.trace import AlignmentResult, RealizabilityReport
from rsi.opd.provenance import build_provenance_report
from rsi.opd.realizability import build_realizability_report


@dataclass(frozen=True)
class AnchorCandidate:
    event_index: int
    action: Action
    source_event_id: str


def find_native_anchors(
    events: Sequence[Event],
    *,
    is_native_action,
) -> list[AnchorCandidate]:
    """Anchor = an event whose action is a Student-native basic action.

    ``is_native_action(action) -> bool`` filters by the H0 basic catalog.
    Internal component events (planning notes, view patches) have no action or
    a non-native action and are skipped automatically.
    """
    out: list[AnchorCandidate] = []
    for i, ev in enumerate(events):
        if ev.event_kind not in (EventKind.TOOL_CALL, EventKind.FINISH, EventKind.MODEL_PROPOSAL):
            continue
        if ev.action is None:
            continue
        if not is_native_action(ev.action):
            continue
        out.append(AnchorCandidate(event_index=i, action=ev.action, source_event_id=ev.event_id))
    return out


def align_to_anchor(
    events: Sequence[Event],
    *,
    student_events: Sequence[Event],
    student_pre_state_digest: str,
    catalog,
    state_view: Mapping[str, object],
    is_native_action,
    uncertain_realizability: str = "skip",
    start_index: int = 0,
) -> AlignmentResult:
    """Skip-to-anchor: first realizable native action at/after ``start_index``.

    Rejects (or converts to acquire-first) anchors whose information is not yet
    realizable in the Student's current visible input.
    """
    anchors = find_native_anchors(events, is_native_action=is_native_action)
    anchors = [a for a in anchors if a.event_index >= start_index]
    if not anchors:
        return AlignmentResult(
            route="skip",
            target_action=None,
            source_event_ids=(),
            anchor_index=None,
            student_pre_state_digest=student_pre_state_digest,
            reject_reason="no_native_anchor",
        )
    student_refs, _ = _split_refs(student_events)
    for anchor in anchors:
        teacher_private_refs = _teacher_private_refs(events[: anchor.event_index + 1])
        legality: RealizabilityReport = build_realizability_report(
            anchor.action,
            catalog=catalog,
            state_view=state_view,
            student_fact_refs=student_refs,
            teacher_private_fact_refs=teacher_private_refs,
        )
        prov = build_provenance_report(anchor.action, student_events=student_events, teacher_events=events)
        if prov.verdict == "leak":
            continue
        if prov.verdict == "uncertain" and uncertain_realizability == "skip":
            continue
        if legality.ok:
            return AlignmentResult(
                route="native_anchor",
                target_action=anchor.action,
                source_event_ids=(anchor.source_event_id,),
                anchor_index=anchor.event_index,
                student_pre_state_digest=student_pre_state_digest,
                legality=legality,
                provenance=prov,
                target_kind="native",
            )
        # Acquire-then-supervise: missing info has a legal acquire action.
        if legality.acquire_first_action is not None:
            return AlignmentResult(
                route="explicit_map",
                target_action=legality.acquire_first_action,
                source_event_ids=(anchor.source_event_id,),
                anchor_index=anchor.event_index,
                student_pre_state_digest=student_pre_state_digest,
                legality=legality,
                provenance=prov,
                target_kind="native",
            )
        # else: skip this anchor, try the next one
    return AlignmentResult(
        route="skip",
        target_action=None,
        source_event_ids=(),
        anchor_index=None,
        student_pre_state_digest=student_pre_state_digest,
        reject_reason="no_realizable_anchor",
    )


def _split_refs(events: Sequence[Event]):
    from rsi.opd.provenance import collect_fact_refs_by_visibility

    return collect_fact_refs_by_visibility(events)


def _teacher_private_refs(events: Sequence[Event]) -> tuple[str, ...]:
    from rsi.contracts.events import Visibility

    refs: set[str] = set()
    for ev in events:
        if ev.visibility != Visibility.TEACHER_PRIVATE:
            continue
        if ev.observation is not None and ev.observation.provenance is not None:
            refs.add(ev.observation.provenance.source_ref)
    return tuple(sorted(refs))
