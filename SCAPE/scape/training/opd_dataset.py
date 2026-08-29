"""Materialize Student-realizable projected training steps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot
from scape.training.action_codec import render_action, validate_roundtrip
from scape.training.opd_events import HarnessEvent
from scape.training.opd_projection import (
    PROJECTION_SCHEMA_VERSION,
    ProjectedAction,
    ProjectionKind,
    ProjectionResult,
    StudentActionSpaceProjector,
)
from scape.training.opd_realizability import (
    apply_student_action,
    check_action_realizability,
    fork_snapshot,
)
from scape.training.tool_mask import legal_tool_names


SNAPSHOT_SCHEMA_VERSION = "scape_snapshot_v2"
TEACHER_ONLY_PROMPT_MARKERS = (
    "teacher_verify_judgment",
    "importance_table",
    "evidence_graph_internal",
    "teacher_only_observation",
    "compressed_teacher_view",
    "VERIFY_RESULT_SECRET",
)


@dataclass
class ProjectedTrainingStep:
    prompt_reduced: str
    target_text: str
    target_action: dict[str, Any]
    token_mask: list[bool] | None
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)
    student_snapshot: dict[str, Any] = field(default_factory=dict)
    source_event_ids: list[str] = field(default_factory=list)
    projection_kind: str = "direct"
    projection_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectionAudit:
    projection_schema_version: str = PROJECTION_SCHEMA_VERSION
    n_teacher_segments: int = 0
    n_direct: int = 0
    n_macro: int = 0
    n_skip_events: int = 0
    n_reject: int = 0
    n_projected_training_steps: int = 0
    n_supervised_tokens: int = 0
    illegal_target_rate: float = 0.0
    inaccessible_reference_rate: float = 0.0
    future_leakage_rate: float = 0.0
    teacher_only_observation_leak_rate: float = 0.0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    mean_anchor_distance: float = 0.0
    mean_macro_length: float = 0.0
    component_id: str | None = None

    @property
    def projection_coverage(self) -> float:
        denom = max(1, self.n_teacher_segments)
        return (self.n_direct + self.n_macro) / denom

    @property
    def direct_rate(self) -> float:
        return self.n_direct / max(1, self.n_teacher_segments)

    @property
    def macro_rate(self) -> float:
        return self.n_macro / max(1, self.n_teacher_segments)

    @property
    def reject_rate(self) -> float:
        return self.n_reject / max(1, self.n_teacher_segments)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "projection_coverage": self.projection_coverage,
                "direct_rate": self.direct_rate,
                "macro_rate": self.macro_rate,
                "reject_rate": self.reject_rate,
            }
        )
        return payload


def render_student_prompt(snapshot: EnvironmentSnapshot, *, component_id: str = "") -> str:
    """Student prefix from reduced harness only. No Teacher-only observations."""
    renderer = DualViewRenderer()
    dual = renderer.render_pair(
        snapshot,
        component_id=component_id or None,
        student_mask=snapshot.harness_mask,
    )
    student_view = dict(dual.student_view)
    for key in TEACHER_ONLY_PROMPT_MARKERS:
        student_view.pop(key, None)
    if not snapshot.harness_mask.get("evidence_graph", False):
        student_view.pop("evidence_graph", None)
    if not snapshot.harness_mask.get("importance_tagging", False):
        student_view.pop("importance", None)
        if isinstance(student_view.get("working_memory"), dict):
            student_view["working_memory"].pop("curated_importance", None)
    if not snapshot.harness_mask.get("verify_tool", False):
        student_view.pop("verify", None)
    # Component context can remain in WorkingMemory snapshots; never expose
    # adaptive rerank instructions when the reduced mask disables the feature.
    if not snapshot.harness_mask.get("adaptive_rerank_instruction", False):
        student_view.pop("rerank_instruction", None)
        if isinstance(student_view.get("working_memory"), dict):
            student_view["working_memory"].pop("rerank_instruction", None)
    return (
        f"System: Harness reduced view (minus {component_id or 'component'}).\n"
        f"Query: {snapshot.query_id}\n"
        f"State:\n{json.dumps(student_view, ensure_ascii=False)}\n"
        f"Assistant:"
    )


def prompt_has_teacher_leak(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(marker.lower() in lowered for marker in TEACHER_ONLY_PROMPT_MARKERS)


def materialize(
    projection: ProjectionResult,
    student_snapshot: EnvironmentSnapshot,
    *,
    component_id: str = "",
) -> list[ProjectedTrainingStep]:
    """Turn an ALIGN/DIRECT Student tool call into training rows.

    SKIP/ε produces no rows. Recovery macros are not materialized.
    """
    if projection.kind != ProjectionKind.DIRECT:
        return []
    cid = component_id or projection.component_id or ""
    mask = student_snapshot.harness_mask
    shadow = fork_snapshot(student_snapshot)
    steps: list[ProjectedTrainingStep] = []
    for action in projection.actions:
        shadow.assert_no_future(max_known_step=shadow.step)
        report = check_action_realizability(
            action=action,
            student_snapshot=shadow,
            student_mask=mask,
            component_id=cid,
        )
        if not report.passed:
            break
        if not validate_roundtrip(action):
            break
        prompt = render_student_prompt(shadow, component_id=cid)
        if prompt_has_teacher_leak(prompt):
            break
        target_text = render_action(action)
        target_action = {
            "name": action.name,
            "arguments": dict(action.arguments),
        }
        steps.append(
            ProjectedTrainingStep(
                prompt_reduced=prompt,
                target_text=target_text,
                target_action=target_action,
                token_mask=None,
                weight=float(action.confidence),
                metadata={
                    "component_id": cid,
                    "realizability": report.to_dict(),
                    "legal_tools": legal_tool_names(harness_mask=mask),
                    "no_teacher_observation_in_student_prefix": True,
                },
                student_snapshot=shadow.to_dict(),
                source_event_ids=list(action.source_event_ids),
                projection_kind=projection.kind.value,
                projection_confidence=float(action.confidence),
            )
        )
        try:
            shadow = apply_student_action(shadow, action)
        except Exception:
            break
    return steps


def build_projected_row(
    *,
    query_id: str,
    component_id: str,
    student_snapshot: EnvironmentSnapshot,
    teacher_events: list[HarnessEvent],
    projection: ProjectionResult,
    projected_steps: list[ProjectedTrainingStep],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "component_id": component_id,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "student_start_snapshot": student_snapshot.to_dict(),
        "student_start_snapshot_hash": student_snapshot.content_hash(),
        "teacher_events": [event.to_dict() for event in teacher_events],
        "projection": projection.to_dict(),
        "projected_steps": [step.to_dict() for step in projected_steps],
        "audit": {
            "no_teacher_observation_in_student_prefix": all(
                step.metadata.get("no_teacher_observation_in_student_prefix") for step in projected_steps
            ),
            "all_targets_student_legal": projection.kind == ProjectionKind.DIRECT
            and bool(projected_steps),
            "realizability_passed": bool(projected_steps)
            and all((step.metadata.get("realizability") or {}).get("passed") for step in projected_steps),
            "reject_reason": projection.reject_reason,
        },
    }


def collect_teacher_shadow_trace(
    *,
    student_snapshot: EnvironmentSnapshot,
    component_id: str,
    events: list[HarnessEvent],
) -> list[HarnessEvent]:
    """Attach a Teacher event list to a student-owned snapshot.

    Production collectors should fork a Teacher shadow from the same snapshot
    and emit Harness events. Tests / smoke pass explicit events.
    """
    del student_snapshot
    out: list[HarnessEvent] = []
    for event in events:
        if event.component_id is None:
            event.component_id = component_id
        out.append(event)
    return out


def project_and_materialize(
    *,
    student_snapshot: EnvironmentSnapshot,
    teacher_events: list[HarnessEvent],
    student_mask: Mapping[str, bool] | None = None,
    component_id: str = "",
    projector: StudentActionSpaceProjector | None = None,
    audit: ProjectionAudit | None = None,
) -> tuple[ProjectionResult, list[ProjectedTrainingStep]]:
    mask = dict(student_mask or student_snapshot.harness_mask)
    proj = projector or StudentActionSpaceProjector()
    trace = collect_teacher_shadow_trace(
        student_snapshot=student_snapshot,
        component_id=component_id,
        events=teacher_events,
    )
    projection = proj.project(
        teacher_trace=trace,
        student_snapshot=student_snapshot,
        student_mask=mask,
    )
    steps: list[ProjectedTrainingStep] = []
    if projection.kind == ProjectionKind.DIRECT:
        steps = materialize(projection, student_snapshot, component_id=component_id)
    if audit is not None:
        audit.n_teacher_segments += 1
        audit.component_id = component_id or projection.component_id
        audit.n_skip_events += len(projection.skipped_event_ids)
        if projection.kind == ProjectionKind.DIRECT:
            audit.n_direct += 1
        else:
            reason = projection.reject_reason or "SKIP"
            audit.reject_reasons[reason] = audit.reject_reasons.get(reason, 0) + 1
        if projection.anchor_distance is not None:
            audit.mean_anchor_distance += float(projection.anchor_distance)
        audit.n_projected_training_steps += len(steps)
    return projection, steps


def finalize_audit(audit: ProjectionAudit) -> ProjectionAudit:
    if audit.n_direct:
        audit.mean_anchor_distance = audit.mean_anchor_distance / max(1, audit.n_direct)
    return audit
