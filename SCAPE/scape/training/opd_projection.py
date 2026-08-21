"""Student-realizable projection: DIRECT / MACRO / SKIP / REJECT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from scape.adapters.components import all_component_ids
from scape.state.snapshot import EnvironmentSnapshot
from scape.training.action_codec import TEACHER_ONLY_TOOLS, canonicalize_action
from scape.training.opd_events import EventKind, HarnessEvent
from scape.training.opd_realizability import (
    RealizabilityReport,
    accessible_doc_ids_of,
    apply_student_action,
    check_action_realizability,
    fork_snapshot,
    has_full_text,
)
from scape.training.tool_mask import legal_tool_names


PROJECTION_SCHEMA_VERSION = "scape_projection_v1"
MAX_ANCHOR_SCAN_EVENTS = 8
MAX_MACRO_ACTIONS = 3

REJECT_ILLEGAL_TOOL = "ILLEGAL_TOOL"
REJECT_INVALID_ARGUMENT_SCHEMA = "INVALID_ARGUMENT_SCHEMA"
REJECT_DOC_NOT_ACCESSIBLE = "DOC_NOT_ACCESSIBLE"
REJECT_TEACHER_ONLY_INFORMATION = "TEACHER_ONLY_INFORMATION"
REJECT_NO_SEMANTIC_ANCHOR = "NO_SEMANTIC_ANCHOR"
REJECT_TRANSITION_NOT_REPRODUCIBLE = "TRANSITION_NOT_REPRODUCIBLE"
REJECT_MACRO_TOO_LONG = "MACRO_TOO_LONG"
REJECT_ANCHOR_HORIZON_EXCEEDED = "ANCHOR_HORIZON_EXCEEDED"
REJECT_STUDENT_EXECUTION_FAILED = "STUDENT_EXECUTION_FAILED"
REJECT_UNKNOWN_COMPONENT_EFFECT = "UNKNOWN_COMPONENT_EFFECT"


class ProjectionKind(str, Enum):
    DIRECT = "direct"
    MACRO = "macro"
    SKIP = "skip"
    REJECT = "reject"


@dataclass
class ProjectedAction:
    name: str
    arguments: dict[str, Any]
    source_event_ids: list[str]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_action({"name": self.name, "arguments": self.arguments}) | {
            "source_event_ids": list(self.source_event_ids),
            "confidence": float(self.confidence),
        }


@dataclass
class ProjectionResult:
    kind: ProjectionKind
    actions: list[ProjectedAction]
    teacher_start_event: str
    teacher_end_event: str
    skipped_event_ids: list[str] = field(default_factory=list)
    anchor_event_id: str | None = None
    anchor_distance: int | None = None
    realizability: RealizabilityReport | None = None
    reject_reason: str | None = None
    component_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "kind": self.kind.value,
            "actions": [a.to_dict() for a in self.actions],
            "teacher_start_event": self.teacher_start_event,
            "teacher_end_event": self.teacher_end_event,
            "skipped_event_ids": list(self.skipped_event_ids),
            "anchor_event_id": self.anchor_event_id,
            "anchor_distance": self.anchor_distance,
            "realizability": None if self.realizability is None else self.realizability.to_dict(),
            "reject_reason": self.reject_reason,
            "component_id": self.component_id,
        }


def _as_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None and str(x)]
    if isinstance(value, dict):
        return [str(x) for x in value.keys()]
    return [str(value)]


def _curate_from_delta(
    added: list[str],
    removed: list[str],
    source_ids: list[str],
) -> ProjectedAction:
    return ProjectedAction(
        name="curate",
        arguments={"add_ids": sorted(added), "remove_ids": sorted(removed)},
        source_event_ids=list(source_ids),
        confidence=1.0,
    )


def _delta_from_event(event: HarnessEvent) -> tuple[list[str], list[str]]:
    delta = event.state_delta or {}
    if "added" in delta or "removed" in delta:
        return _as_ids(delta.get("added")), _as_ids(delta.get("removed"))
    before = set(_as_ids(delta.get("before_curated") or delta.get("curated_before")))
    after = set(_as_ids(delta.get("after_curated") or delta.get("curated_after")))
    if before or after:
        return sorted(after - before), sorted(before - after)
    return [], []


def _component_of(events: list[HarnessEvent], snapshot: EnvironmentSnapshot) -> str:
    for event in events:
        if event.component_id:
            return str(event.component_id)
    return str(snapshot.metadata.get("component_id") or "")


Handler = Callable[
    [list[HarnessEvent], int, EnvironmentSnapshot, Mapping[str, bool], "StudentActionSpaceProjector"],
    ProjectionResult,
]


class StudentActionSpaceProjector:
    """Compile Full-Harness Teacher events into Student-legal actions."""

    def __init__(
        self,
        *,
        max_anchor_scan_events: int = MAX_ANCHOR_SCAN_EVENTS,
        max_macro_actions: int = MAX_MACRO_ACTIONS,
    ) -> None:
        self.max_anchor_scan_events = int(max_anchor_scan_events)
        self.max_macro_actions = int(max_macro_actions)
        self.handlers: dict[str, Handler] = {
            "auto_populate_first_search": self._handle_auto_populate,
            "subtractive_curation": self._handle_subtractive,
            "importance_tagging": self._handle_importance,
            "evidence_graph": self._handle_skip_to_anchor,
            "sentence_compress": self._handle_skip_to_anchor,
            "content_dedup": self._handle_content_dedup,
            "verify_tool": self._handle_verify,
            "token_budget_marker": self._handle_token_budget,
            "adaptive_rerank_instruction": self._handle_adaptive_rerank,
            "chunk_neighbors": self._handle_chunk_neighbors,
        }
        for cid in all_component_ids():
            self.handlers.setdefault(cid, self._handle_unknown)

    def project(
        self,
        *,
        teacher_trace: list[HarnessEvent],
        student_snapshot: EnvironmentSnapshot,
        student_mask: Mapping[str, bool],
    ) -> ProjectionResult:
        return self.project_segment(
            teacher_events=teacher_trace,
            start_index=0,
            student_snapshot=student_snapshot,
            student_mask=student_mask,
        )

    def project_segment(
        self,
        *,
        teacher_events: list[HarnessEvent],
        start_index: int,
        student_snapshot: EnvironmentSnapshot,
        student_mask: Mapping[str, bool],
    ) -> ProjectionResult:
        events = list(teacher_events)
        if not events or start_index >= len(events):
            return self._reject(
                events,
                start_index,
                REJECT_NO_SEMANTIC_ANCHOR,
                component_id=_component_of(events, student_snapshot),
            )
        component_id = _component_of(events[start_index:], student_snapshot)
        handler = self.handlers.get(component_id, self._handle_unknown)
        result = handler(events, start_index, student_snapshot, student_mask, self)
        result.component_id = component_id or result.component_id
        return self._finalize(result, student_snapshot, student_mask, component_id)

    def _finalize(
        self,
        result: ProjectionResult,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        component_id: str,
    ) -> ProjectionResult:
        if result.kind in {ProjectionKind.SKIP, ProjectionKind.REJECT}:
            return result
        if len(result.actions) > self.max_macro_actions:
            result.kind = ProjectionKind.REJECT
            result.reject_reason = REJECT_MACRO_TOO_LONG
            result.actions = []
            return result
        legal = set(legal_tool_names(harness_mask=mask))
        reports: list[RealizabilityReport] = []
        shadow = fork_snapshot(snapshot)
        for action in result.actions:
            if action.name not in legal or action.name in TEACHER_ONLY_TOOLS:
                result.kind = ProjectionKind.REJECT
                result.reject_reason = REJECT_ILLEGAL_TOOL
                result.actions = []
                return result
            if action.name == "curate" and "importance" in action.arguments:
                result.kind = ProjectionKind.REJECT
                result.reject_reason = REJECT_INVALID_ARGUMENT_SCHEMA
                result.actions = []
                return result
            report = check_action_realizability(
                action=action,
                student_snapshot=shadow,
                student_mask=mask,
                component_id=component_id,
            )
            reports.append(report)
            if not report.passed:
                result.kind = ProjectionKind.REJECT
                result.reject_reason = report.reason_codes[0] if report.reason_codes else REJECT_NO_SEMANTIC_ANCHOR
                result.realizability = report
                result.actions = []
                return result
            try:
                shadow = apply_student_action(shadow, action)
            except Exception:
                result.kind = ProjectionKind.REJECT
                result.reject_reason = REJECT_STUDENT_EXECUTION_FAILED
                result.actions = []
                return result
        if reports:
            result.realizability = reports[0]
        if len(result.actions) > 1:
            result.kind = ProjectionKind.MACRO
        elif result.kind != ProjectionKind.MACRO:
            result.kind = ProjectionKind.DIRECT
        return result

    def _reject(
        self,
        events: list[HarnessEvent],
        start_index: int,
        reason: str,
        *,
        skipped: list[str] | None = None,
        component_id: str | None = None,
    ) -> ProjectionResult:
        start = events[start_index] if events and start_index < len(events) else None
        end = events[-1] if events else None
        return ProjectionResult(
            kind=ProjectionKind.REJECT,
            actions=[],
            teacher_start_event="" if start is None else start.event_id,
            teacher_end_event="" if end is None else end.event_id,
            skipped_event_ids=list(skipped or []),
            reject_reason=reason,
            component_id=component_id,
        )

    def _skip_only(
        self,
        events: list[HarnessEvent],
        start_index: int,
        skipped: list[str],
        component_id: str,
    ) -> ProjectionResult:
        start = events[start_index]
        return ProjectionResult(
            kind=ProjectionKind.SKIP,
            actions=[],
            teacher_start_event=start.event_id,
            teacher_end_event=events[min(len(events) - 1, start_index + len(skipped))].event_id,
            skipped_event_ids=list(skipped),
            component_id=component_id,
        )

    def _ok(
        self,
        kind: ProjectionKind,
        actions: list[ProjectedAction],
        events: list[HarnessEvent],
        start_index: int,
        end_index: int,
        *,
        skipped: list[str] | None = None,
        component_id: str | None = None,
    ) -> ProjectionResult:
        start = events[start_index]
        end = events[end_index]
        return ProjectionResult(
            kind=kind,
            actions=actions,
            teacher_start_event=start.event_id,
            teacher_end_event=end.event_id,
            skipped_event_ids=list(skipped or []),
            anchor_event_id=end.event_id,
            anchor_distance=max(0, end_index - start_index),
            component_id=component_id,
        )

    def _scan_downstream_action(
        self,
        events: list[HarnessEvent],
        start_index: int,
        skip_kinds: set[EventKind],
        skip_names: set[str],
    ) -> tuple[list[str], HarnessEvent | None, int]:
        skipped: list[str] = []
        last = min(len(events), start_index + self.max_anchor_scan_events)
        for idx in range(start_index, last):
            event = events[idx]
            if event.kind in skip_kinds or (
                event.kind == EventKind.MODEL_ACTION and event.action_name in skip_names
            ):
                skipped.append(event.event_id)
                continue
            if event.kind == EventKind.TOOL_OBSERVATION and not event.visible_to_student:
                skipped.append(event.event_id)
                continue
            if event.kind == EventKind.MODEL_ACTION and event.action_name:
                return skipped, event, idx
            skipped.append(event.event_id)
        return skipped, None, last

    def _handle_curated_delta(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        component_id: str,
        *,
        skip_if_empty: bool,
    ) -> ProjectionResult:
        del mask
        skipped: list[str] = []
        last = min(len(events), start_index + self.max_anchor_scan_events)
        for idx in range(start_index, last):
            event = events[idx]
            added, removed = _delta_from_event(event)
            if not added and not removed:
                skipped.append(event.event_id)
                continue
            accessible = set(accessible_doc_ids_of(snapshot))
            if any(doc not in accessible for doc in added):
                return self._reject(
                    events, start_index, REJECT_DOC_NOT_ACCESSIBLE, skipped=skipped, component_id=component_id
                )
            action = _curate_from_delta(added, removed, [event.event_id])
            return self._ok(
                ProjectionKind.MACRO if event.kind != EventKind.MODEL_ACTION or event.action_name != "curate" else ProjectionKind.DIRECT,
                [action],
                events,
                start_index,
                idx,
                skipped=skipped,
                component_id=component_id,
            )
        if skip_if_empty and skipped:
            return self._skip_only(events, start_index, skipped, component_id)
        return self._reject(
            events, start_index, REJECT_NO_SEMANTIC_ANCHOR, skipped=skipped, component_id=component_id
        )

    def _handle_auto_populate(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        return self._handle_curated_delta(
            events, start_index, snapshot, mask, "auto_populate_first_search", skip_if_empty=False
        )

    def _handle_subtractive(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        return self._handle_curated_delta(
            events, start_index, snapshot, mask, "subtractive_curation", skip_if_empty=False
        )

    def _handle_importance(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        # Latent importance mutation itself is SKIP; only curated-set change is distilled.
        return self._handle_curated_delta(
            events, start_index, snapshot, mask, "importance_tagging", skip_if_empty=True
        )

    def _direct_or_reject_downstream(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        component_id: str,
        skip_kinds: set[EventKind],
        skip_names: set[str],
    ) -> ProjectionResult:
        skipped, event, idx = self._scan_downstream_action(events, start_index, skip_kinds, skip_names)
        if event is None:
            if idx - start_index >= self.max_anchor_scan_events:
                return self._reject(
                    events, start_index, REJECT_ANCHOR_HORIZON_EXCEEDED, skipped=skipped, component_id=component_id
                )
            if skipped:
                return self._skip_only(events, start_index, skipped, component_id)
            return self._reject(
                events, start_index, REJECT_NO_SEMANTIC_ANCHOR, skipped=skipped, component_id=component_id
            )
        if event.action_name in TEACHER_ONLY_TOOLS:
            return self._reject(
                events, start_index, REJECT_TEACHER_ONLY_INFORMATION, skipped=skipped, component_id=component_id
            )
        action = ProjectedAction(
            name=str(event.action_name),
            arguments=dict(event.arguments or {}),
            source_event_ids=[event.event_id],
        )
        report = check_action_realizability(
            action=action, student_snapshot=snapshot, student_mask=mask, component_id=component_id
        )
        if not report.passed:
            return self._reject(
                events,
                start_index,
                report.reason_codes[0] if report.reason_codes else REJECT_NO_SEMANTIC_ANCHOR,
                skipped=skipped,
                component_id=component_id,
            )
        return self._ok(
            ProjectionKind.DIRECT,
            [action],
            events,
            start_index,
            idx,
            skipped=skipped,
            component_id=component_id,
        )

    def _handle_skip_to_anchor(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        component_id = _component_of(events[start_index:], snapshot) or "evidence_graph"
        return self._direct_or_reject_downstream(
            events,
            start_index,
            snapshot,
            mask,
            component_id,
            skip_kinds={EventKind.HARNESS_MUTATION, EventKind.OBS_TRANSFORM},
            skip_names=set(),
        )

    def _handle_content_dedup(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        # Extra Student duplicates are fine. Id remapping without a verified map is REJECT.
        for event in events[start_index : start_index + self.max_anchor_scan_events]:
            mapping = (event.state_delta or {}).get("duplicate_equivalence") or event.metadata.get(
                "duplicate_equivalence"
            )
            remapped = (event.state_delta or {}).get("canonical_id_changed")
            if remapped and not mapping:
                return self._reject(
                    events, start_index, REJECT_TRANSITION_NOT_REPRODUCIBLE, component_id="content_dedup"
                )
        return self._direct_or_reject_downstream(
            events,
            start_index,
            snapshot,
            mask,
            "content_dedup",
            skip_kinds={EventKind.HARNESS_MUTATION, EventKind.OBS_TRANSFORM},
            skip_names=set(),
        )

    def _handle_verify(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        skipped: list[str] = []
        last = min(len(events), start_index + self.max_anchor_scan_events)
        verify_docs: list[str] = []
        saw_verify = False
        for idx in range(start_index, last):
            event = events[idx]
            if event.kind == EventKind.MODEL_ACTION and event.action_name == "verify":
                saw_verify = True
                verify_docs = _as_ids(
                    event.arguments.get("evidence_ids")
                    or event.arguments.get("doc_ids")
                    or event.arguments.get("doc_id")
                )
                skipped.append(event.event_id)
                continue
            if event.kind == EventKind.TOOL_OBSERVATION and not event.visible_to_student:
                skipped.append(event.event_id)
                continue
            if event.kind == EventKind.MODEL_ACTION and event.action_name:
                if event.action_name == "verify":
                    skipped.append(event.event_id)
                    continue
                docs = verify_docs or _as_ids(
                    event.arguments.get("add_ids")
                    or event.arguments.get("doc_ids")
                    or event.arguments.get("doc_id")
                )
                accessible = set(accessible_doc_ids_of(snapshot))
                if any(doc not in accessible for doc in docs):
                    return self._reject(
                        events,
                        start_index,
                        REJECT_TEACHER_ONLY_INFORMATION,
                        skipped=skipped,
                        component_id="verify_tool",
                    )
                if saw_verify and docs and all(has_full_text(snapshot, doc) for doc in docs):
                    action = ProjectedAction(
                        name=str(event.action_name),
                        arguments=dict(event.arguments or {}),
                        source_event_ids=[event.event_id],
                    )
                    return self._ok(
                        ProjectionKind.DIRECT,
                        [action],
                        events,
                        start_index,
                        idx,
                        skipped=skipped,
                        component_id="verify_tool",
                    )
                if saw_verify and docs and all(doc in accessible for doc in docs):
                    review = ProjectedAction(
                        name="review_docs",
                        arguments={"doc_ids": sorted(docs)},
                        source_event_ids=[events[start_index].event_id],
                    )
                    downstream = ProjectedAction(
                        name=str(event.action_name),
                        arguments=dict(event.arguments or {}),
                        source_event_ids=[event.event_id],
                    )
                    return self._ok(
                        ProjectionKind.MACRO,
                        [review, downstream],
                        events,
                        start_index,
                        idx,
                        skipped=skipped,
                        component_id="verify_tool",
                    )
                if not saw_verify:
                    action = ProjectedAction(
                        name=str(event.action_name),
                        arguments=dict(event.arguments or {}),
                        source_event_ids=[event.event_id],
                    )
                    return self._ok(
                        ProjectionKind.DIRECT,
                        [action],
                        events,
                        start_index,
                        idx,
                        skipped=skipped,
                        component_id="verify_tool",
                    )
                return self._reject(
                    events,
                    start_index,
                    REJECT_TEACHER_ONLY_INFORMATION,
                    skipped=skipped,
                    component_id="verify_tool",
                )
            skipped.append(event.event_id)
        if skipped and not any(
            ev.kind == EventKind.MODEL_ACTION and ev.action_name not in TEACHER_ONLY_TOOLS
            for ev in events[start_index:last]
        ):
            return self._skip_only(events, start_index, skipped, "verify_tool")
        return self._reject(
            events, start_index, REJECT_TEACHER_ONLY_INFORMATION, skipped=skipped, component_id="verify_tool"
        )

    def _handle_token_budget(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        event = events[start_index]
        if event.metadata.get("requires_external_accounting") or (event.state_delta or {}).get(
            "requires_external_accounting"
        ):
            return self._reject(
                events, start_index, REJECT_TEACHER_ONLY_INFORMATION, component_id="token_budget_marker"
            )
        return self._direct_or_reject_downstream(
            events,
            start_index,
            snapshot,
            mask,
            "token_budget_marker",
            skip_kinds={EventKind.HARNESS_MUTATION, EventKind.OBS_TRANSFORM},
            skip_names=set(),
        )

    def _handle_adaptive_rerank(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector, mask
        skipped: list[str] = []
        last = min(len(events), start_index + self.max_anchor_scan_events)
        for idx in range(start_index, last):
            event = events[idx]
            if event.kind != EventKind.MODEL_ACTION or event.action_name not in {
                "search_corpus",
                "fan_out_search",
                "grep_corpus",
            }:
                if event.kind in {EventKind.HARNESS_MUTATION, EventKind.OBS_TRANSFORM, EventKind.TOOL_OBSERVATION}:
                    skipped.append(event.event_id)
                    continue
                if event.kind == EventKind.MODEL_ACTION and event.action_name:
                    action = ProjectedAction(
                        name=str(event.action_name),
                        arguments=dict(event.arguments or {}),
                        source_event_ids=[event.event_id],
                    )
                    return self._ok(
                        ProjectionKind.DIRECT,
                        [action],
                        events,
                        start_index,
                        idx,
                        skipped=skipped,
                        component_id="adaptive_rerank_instruction",
                    )
                skipped.append(event.event_id)
                continue
            teacher_hits = set(
                _as_ids(
                    (event.state_delta or {}).get("result_ids")
                    or event.metadata.get("teacher_result_ids")
                    or event.arguments.get("result_ids")
                )
            )
            student_hits = set(
                _as_ids(
                    snapshot.metadata.get("student_search_results")
                    or snapshot.working_memory.get("student_search_results")
                )
            )
            needed = set(_as_ids(snapshot.metadata.get("teacher_needed_doc_ids") or teacher_hits))
            accessible = set(accessible_doc_ids_of(snapshot))
            if needed and needed.issubset(student_hits | accessible):
                action = ProjectedAction(
                    name=str(event.action_name),
                    arguments={k: v for k, v in (event.arguments or {}).items() if k != "result_ids"},
                    source_event_ids=[event.event_id],
                )
                return self._ok(
                    ProjectionKind.DIRECT,
                    [action],
                    events,
                    start_index,
                    idx,
                    skipped=skipped,
                    component_id="adaptive_rerank_instruction",
                )
            # First version: do not invent query rewrites.
            return self._reject(
                events,
                start_index,
                REJECT_TRANSITION_NOT_REPRODUCIBLE,
                skipped=skipped,
                component_id="adaptive_rerank_instruction",
            )
        return self._reject(
            events,
            start_index,
            REJECT_ANCHOR_HORIZON_EXCEEDED,
            skipped=skipped,
            component_id="adaptive_rerank_instruction",
        )

    def _handle_chunk_neighbors(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector
        # Neighbor expansion itself is SKIP. Recover via read/review when possible.
        skipped, event, idx = self._scan_downstream_action(
            events,
            start_index,
            {EventKind.HARNESS_MUTATION, EventKind.OBS_TRANSFORM},
            set(),
        )
        if event is None:
            if skipped:
                return self._skip_only(events, start_index, skipped, "chunk_neighbors")
            return self._reject(
                events, start_index, REJECT_NO_SEMANTIC_ANCHOR, skipped=skipped, component_id="chunk_neighbors"
            )
        needed = _as_ids((event.state_delta or {}).get("neighbor_ids") or event.arguments.get("doc_ids"))
        accessible = set(accessible_doc_ids_of(snapshot))
        missing = [doc for doc in needed if doc not in accessible]
        if not missing:
            action = ProjectedAction(
                name=str(event.action_name),
                arguments=dict(event.arguments or {}),
                source_event_ids=[event.event_id],
            )
            return self._ok(
                ProjectionKind.DIRECT,
                [action],
                events,
                start_index,
                idx,
                skipped=skipped,
                component_id="chunk_neighbors",
            )
        if len(missing) <= self.max_macro_actions:
            macros = [
                ProjectedAction(
                    name="read_document",
                    arguments={"doc_id": doc},
                    source_event_ids=[event.event_id],
                )
                for doc in missing
            ]
            macros.append(
                ProjectedAction(
                    name=str(event.action_name),
                    arguments=dict(event.arguments or {}),
                    source_event_ids=[event.event_id],
                )
            )
            if len(macros) > self.max_macro_actions:
                return self._reject(
                    events, start_index, REJECT_MACRO_TOO_LONG, skipped=skipped, component_id="chunk_neighbors"
                )
            return self._ok(
                ProjectionKind.MACRO,
                macros,
                events,
                start_index,
                idx,
                skipped=skipped,
                component_id="chunk_neighbors",
            )
        return self._reject(
            events, start_index, REJECT_DOC_NOT_ACCESSIBLE, skipped=skipped, component_id="chunk_neighbors"
        )

    def _handle_unknown(
        self,
        events: list[HarnessEvent],
        start_index: int,
        snapshot: EnvironmentSnapshot,
        mask: Mapping[str, bool],
        projector: "StudentActionSpaceProjector",
    ) -> ProjectionResult:
        del projector, mask
        return self._reject(
            events,
            start_index,
            REJECT_UNKNOWN_COMPONENT_EFFECT,
            component_id=_component_of(events[start_index:], snapshot),
        )
