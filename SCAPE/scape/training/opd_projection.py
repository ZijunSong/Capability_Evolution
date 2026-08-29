"""Student-realizable projection: ALIGN (student tool call) or SKIP (ε).

Canonical algorithm lives in EasyOPD's skip-to-anchor projector, which
wraps Harness-1 rather than reimplementing component semantics here.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from scape.adapters.components import all_component_ids
from scape.state.snapshot import EnvironmentSnapshot
from scape.training.action_codec import canonicalize_action
from scape.training.opd_events import HarnessEvent
from scape.training.opd_realizability import RealizabilityReport, accessible_doc_ids_of
from scape.training.tool_mask import legal_tool_names

PROJECTION_SCHEMA_VERSION = "scape_projection_v1"
MAX_ANCHOR_SCAN_EVENTS = 8
MAX_MACRO_ACTIONS = 1  # kept for call-site compatibility; macros are not emitted

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


def _load_skip_to_anchor():
    path = (
        Path(__file__).resolve().parents[3]
        / "SCAPE-EasyOPD"
        / "easyopd"
        / "methods"
        / "scape_component_opd"
        / "skip_to_anchor.py"
    )
    spec = importlib.util.spec_from_file_location("scape_easyopd_skip_to_anchor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load EasyOPD skip-to-anchor projector from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STA = _load_skip_to_anchor()


class ProjectionKind(str, Enum):
    DIRECT = "direct"  # aligned Student tool call
    SKIP = "skip"  # ε: no Student action; scan continued
    MACRO = "macro"  # unused: recovery macros are not emitted
    REJECT = "reject"  # unused as a Teacher action; unrealizable events SKIP


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


def _event_to_dict(event: HarnessEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "kind": event.kind.value if hasattr(event.kind, "value") else str(event.kind),
        "event_type": (event.metadata or {}).get("event_type"),
        "action_name": event.action_name,
        "arguments": dict(event.arguments or {}),
        "state_delta": dict(event.state_delta or {}),
        "payload": dict(event.metadata or {}),
        "projectable_target": (event.metadata or {}).get("projectable_target"),
        "visible_to_student": bool(event.visible_to_student),
        "component_id": event.component_id,
    }


def _from_skip_to_anchor(raw: Any, component_id: str) -> ProjectionResult:
    if raw.kind not in {_STA.ALIGN, _STA.SKIP}:
        raise ValueError(f"EasyOPD projector must emit align or skip, got {raw.kind}")
    kind = ProjectionKind.DIRECT if raw.kind == _STA.ALIGN else ProjectionKind.SKIP
    actions = [
        ProjectedAction(
            name=action.name,
            arguments=dict(action.arguments),
            source_event_ids=[action.source_event_id],
            confidence=float(action.confidence),
        )
        for action in raw.actions
    ]
    return ProjectionResult(
        kind=kind,
        actions=actions,
        teacher_start_event=raw.teacher_start_event,
        teacher_end_event=raw.teacher_end_event,
        skipped_event_ids=list(raw.skipped_event_ids),
        anchor_event_id=raw.anchor_event_id,
        anchor_distance=raw.anchor_distance,
        reject_reason=raw.reason,
        component_id=component_id or raw.component_id,
    )


class StudentActionSpaceProjector:
    """Compile Full-Harness Teacher events into Student-legal actions via EasyOPD."""

    def __init__(
        self,
        *,
        max_anchor_scan_events: int = MAX_ANCHOR_SCAN_EVENTS,
        max_macro_actions: int = MAX_MACRO_ACTIONS,
    ) -> None:
        self.max_anchor_scan_events = int(max_anchor_scan_events)
        self.max_macro_actions = int(max_macro_actions)
        self.handlers = {cid: self.project_segment for cid in all_component_ids()}

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
        projector: "StudentActionSpaceProjector" | None = None,
    ) -> ProjectionResult:
        del projector
        events = list(teacher_events)
        component_id = ""
        for event in events[start_index:]:
            if event.component_id:
                component_id = str(event.component_id)
                break
        component_id = component_id or str(student_snapshot.metadata.get("component_id") or "")
        if not events or start_index >= len(events):
            return ProjectionResult(
                kind=ProjectionKind.SKIP,
                actions=[],
                teacher_start_event="",
                teacher_end_event="",
                reject_reason=REJECT_NO_SEMANTIC_ANCHOR,
                component_id=component_id,
            )
        payload = [_event_to_dict(event) for event in events[start_index:]]
        results = _STA.project_events(
            payload,
            accessible_doc_ids=accessible_doc_ids_of(student_snapshot),
            legal_tools=legal_tool_names(harness_mask=student_mask),
            student_search_results=student_snapshot.metadata.get("student_search_results"),
            teacher_needed_doc_ids=student_snapshot.metadata.get("teacher_needed_doc_ids"),
            max_anchor_scan_events=self.max_anchor_scan_events,
            component_id=component_id,
            first_only=True,
        )
        return _from_skip_to_anchor(results[0], component_id)
