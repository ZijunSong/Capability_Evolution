"""InformationSafeGate visibility tests (Case 3)."""

from __future__ import annotations

from harness.artifacts.gates import run_information_safe_gates, visibility_gate
from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from training.scope.routing import route_decision
from training.scope.schema import Route

from tests.scope.conftest import make_state


def test_visibility_gate_passes_observed_evidence():
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    art = PrivilegedArtifact.build(
        episode_id="ep1",
        turn_id=3,
        module_id="evidence_state",
        mode=GuidanceMode.CORRECT,
        reason_code="DUPLICATE_EVIDENCE",
        student_action=student,
        evidence_ids=("obs_1",),
        document_ids=("d1",),
        capability_id="duplicate_evidence",
        recommended_operation="skip_curate",
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": []},
        ),
    )
    gate = visibility_gate(state, art)
    assert gate.passed


def test_case3_visibility_fail_routes_ignore():
    """Case 3: artifact cites obs_999 not seen by student → IGNORE."""
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    art = PrivilegedArtifact.build(
        episode_id="ep1",
        turn_id=3,
        module_id="evidence_state",
        mode=GuidanceMode.CORRECT,
        reason_code="DUPLICATE_EVIDENCE",
        student_action=student,
        evidence_ids=("obs_999",),
        document_ids=("d1",),
        capability_id="duplicate_evidence",
        recommended_operation="skip_curate",
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": []},
        ),
    )
    report = run_information_safe_gates(state, art)
    assert not report.visible
    result = route_decision(state, art, student)
    assert result.route == Route.IGNORE
    assert result.sample.train_mask == 0
