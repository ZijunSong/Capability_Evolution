"""Verified Decision Routing: student + artifact + verifier → target action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.artifacts.gates import (
    GateResult,
    InformationSafeReport,
    run_information_safe_gates,
)
from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.validators import ValidationResult, get_verifier
from harness.capability.action_space import CapabilityAction
from harness.capability.capability_id import is_round1_trainable
from harness.capability.distillability import round1_purity
from harness.capability.state import DecisionState
from harness.shadow.action_realizer import ActionRealizer, CandidateAction, realize
from training.scope.schema import (
    BranchType,
    DecisionSupervisionSampleV3,
    GateFlags,
    Route,
    VerificationFlags,
    WeightTerms,
)


@dataclass(frozen=True)
class RoutingResult:
    route: Route
    target_action: CapabilityAction | None
    candidate: CandidateAction | None
    gates: InformationSafeReport
    student_validation: ValidationResult | None
    target_validation: ValidationResult | None
    sample: DecisionSupervisionSampleV3

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "target_action": self.target_action.to_dict() if self.target_action else None,
            "gates": self.gates.to_dict(),
            "sample": self.sample.to_dict(),
        }


def _gate_flags(report: InformationSafeReport) -> GateFlags:
    return GateFlags(
        visible=report.visible,
        schema_valid=report.schema_valid,
        module_valid=report.module_valid,
        executable=report.executable,
        provenance_ok=report.provenance_ok,
        purity_ok=report.purity_ok,
    )


def route_decision(
    state: DecisionState,
    artifact: PrivilegedArtifact,
    student_action: CapabilityAction | None = None,
    *,
    realizer: ActionRealizer | None = None,
    fingerprint_before: dict[str, Any] | None = None,
    fingerprint_after: dict[str, Any] | None = None,
    branch_type: BranchType = BranchType.MAIN,
    event_id: str = "",
    enforce_round1_capability_filter: bool = True,
    student_state_text: str = "",
    render_action=None,
) -> RoutingResult:
    """Endorse / Correct / Ignore → unified DecisionSupervisionSampleV3."""
    student = student_action or artifact.student_action
    verifier = get_verifier(artifact.module_id)
    realizer = realizer or ActionRealizer()

    # Capability filter for Round 1
    cap = artifact.resolved_capability()
    if enforce_round1_capability_filter and not is_round1_trainable(cap):
        gates = run_information_safe_gates(
            state,
            artifact,
            fingerprint_before=fingerprint_before,
            fingerprint_after=fingerprint_after,
        )
        sample = DecisionSupervisionSampleV3.build(
            state=state,
            artifact=artifact,
            student_action=student,
            target_action=None,
            route=Route.IGNORE,
            gates=_gate_flags(gates),
            train_mask=0,
            audit_error="CAPABILITY_DISABLED_ROUND1",
            branch_type=branch_type,
            event_id=event_id,
            student_state_text=student_state_text,
            metadata={"capability_filter": "round1"},
        )
        return RoutingResult(
            route=Route.IGNORE,
            target_action=None,
            candidate=None,
            gates=gates,
            student_validation=None,
            target_validation=None,
            sample=sample,
        )

    student_val = verifier.validate(state, student, artifact)
    candidate: CandidateAction | None = None
    target: CapabilityAction | None = None
    target_val: ValidationResult | None = None
    route = Route.IGNORE

    if artifact.mode == GuidanceMode.ENDORSE and student_val.valid:
        target = student
        target_val = student_val
        route = Route.ENDORSE
    elif artifact.mode == GuidanceMode.CORRECT:
        candidate = realizer.realize(state, artifact)
        if candidate is not None:
            gates_pre = run_information_safe_gates(
                state,
                artifact,
                candidate_action=candidate.action,
                fingerprint_before=fingerprint_before,
                fingerprint_after=fingerprint_after,
            )
            target_val = verifier.validate(state, candidate.action, artifact)
            if gates_pre.all_passed and target_val.valid:
                target = candidate.action
                route = Route.CORRECT
            else:
                route = Route.IGNORE
        else:
            route = Route.IGNORE
    else:
        route = Route.IGNORE

    gates = run_information_safe_gates(
        state,
        artifact,
        candidate_action=target,
        fingerprint_before=fingerprint_before,
        fingerprint_after=fingerprint_after,
    )

    # If gates fail after routing, force IGNORE
    audit_error = gates.audit_error
    train_mask = 1
    if route != Route.IGNORE and not gates.all_passed:
        route = Route.IGNORE
        target = None
        train_mask = 0
        audit_error = audit_error or "GATE_FAILED"
    if route == Route.IGNORE:
        train_mask = 0 if audit_error else train_mask
        if target is None:
            train_mask = 0

    purity = round1_purity(cap)
    weight_terms = WeightTerms(
        procedural_purity=purity.purity_score,
        reliability=1.0 if (target_val and target_val.valid) else 0.0,
        internalization=0.0,
        local_gain=1.0 if route in {Route.ENDORSE, Route.CORRECT} else 0.0,
    )
    # Round 1: uniform sample weight (no adaptive)
    sample_weight = 1.0 if route in {Route.ENDORSE, Route.CORRECT} and train_mask else 0.0

    target_text = ""
    if target is not None and render_action is not None:
        target_text = render_action(target)
    elif target is not None:
        target_text = target.canonical_key()

    sample = DecisionSupervisionSampleV3.build(
        state=state,
        artifact=artifact,
        student_action=student,
        target_action=target,
        route=route,
        gates=_gate_flags(gates),
        verification=VerificationFlags(
            student_valid=student_val.valid,
            target_valid=target_val.valid if target_val else None,
            score_student=student_val.score,
            score_target=target_val.score if target_val else None,
        ),
        weight_terms=weight_terms,
        sample_weight=sample_weight,
        branch_type=branch_type,
        event_id=event_id,
        train_mask=train_mask if route != Route.IGNORE else 0,
        student_state_text=student_state_text,
        target_action_text=target_text,
        audit_error=audit_error,
        metadata={
            "guidance_mode": artifact.mode.value,
            "candidate_source": candidate.source if candidate else None,
            "purity": purity.to_dict(),
        },
    )
    return RoutingResult(
        route=route,
        target_action=target,
        candidate=candidate,
        gates=gates,
        student_validation=student_val,
        target_validation=target_val,
        sample=sample,
    )
