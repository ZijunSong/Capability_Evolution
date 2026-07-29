"""End-to-end SCOPE v3 supervision pipeline.

DecisionStateV2 → LocalDecisionArtifactV3 → InformationSafeGate
    → ActionRealizer → VerifiedDecisionRouting → DecisionSupervisionSampleV3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from harness.artifacts.gates import capture_env_fingerprint, run_information_safe_gates
from harness.artifacts.schema import PrivilegedArtifact
from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState
from harness.shadow.base import ShadowModule
from harness.telemetry.writer import ScopeTelemetryWriter
from training.scope.routing import RoutingResult, route_decision
from training.scope.schema import BranchType, DecisionSupervisionSampleV3


@dataclass(frozen=True)
class SupervisionPipelineResult:
    """Full v3 chain output for one decision point."""

    state: DecisionState
    student_action: CapabilityAction
    artifact: PrivilegedArtifact
    routing: RoutingResult
    fingerprint_before: dict[str, Any] | None
    fingerprint_after: dict[str, Any] | None

    @property
    def sample(self) -> DecisionSupervisionSampleV3:
        return self.routing.sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "student_action": self.student_action.to_dict(),
            "artifact": self.artifact.to_dict(),
            "routing": self.routing.to_dict(),
            "fingerprint_before": self.fingerprint_before,
            "fingerprint_after": self.fingerprint_after,
        }


def run_supervision_pipeline(
    state: DecisionState,
    student_action: CapabilityAction,
    *,
    artifact: PrivilegedArtifact | None = None,
    shadow: ShadowModule | None = None,
    env: Any | None = None,
    fingerprint_before: dict[str, Any] | None = None,
    fingerprint_after: dict[str, Any] | None = None,
    branch_type: BranchType = BranchType.MAIN,
    event_id: str = "",
    student_state_text: str = "",
    telemetry: ScopeTelemetryWriter | None = None,
    render_action: Callable[[CapabilityAction], str] | None = None,
    enforce_round1_capability_filter: bool = True,
) -> SupervisionPipelineResult:
    """Run shadow → gates → realizer → routing and optionally emit telemetry."""
    if artifact is None:
        if shadow is None:
            raise ValueError("run_supervision_pipeline requires artifact or shadow")
        fp_before = fingerprint_before
        if fp_before is None and env is not None:
            fp_before = capture_env_fingerprint(env)
        artifact = shadow.analyze(state, student_action)
        fp_after = fingerprint_after
        if fp_after is None and env is not None:
            fp_after = capture_env_fingerprint(env)
        fingerprint_before = fp_before
        fingerprint_after = fp_after
    else:
        if fingerprint_before is None and env is not None:
            fingerprint_before = capture_env_fingerprint(env)
        if fingerprint_after is None and env is not None:
            fingerprint_after = capture_env_fingerprint(env)

    # Pre-routing gate snapshot (artifact-only, no candidate yet)
    pre_gates = run_information_safe_gates(
        state,
        artifact,
        fingerprint_before=fingerprint_before,
        fingerprint_after=fingerprint_after,
    )

    routing = route_decision(
        state,
        artifact,
        student_action,
        fingerprint_before=fingerprint_before,
        fingerprint_after=fingerprint_after,
        branch_type=branch_type,
        event_id=event_id or state.event_id,
        student_state_text=student_state_text or state.rendered_context,
        enforce_round1_capability_filter=enforce_round1_capability_filter,
        render_action=render_action,
    )

    result = SupervisionPipelineResult(
        state=state,
        student_action=student_action,
        artifact=artifact,
        routing=routing,
        fingerprint_before=fingerprint_before,
        fingerprint_after=fingerprint_after,
    )

    if telemetry is not None:
        telemetry.record_supervision_pipeline(
            state=state,
            student_action=student_action,
            artifact=artifact,
            routing=routing,
            pre_gates=pre_gates,
            event_id=event_id or state.event_id,
        )

    return result
