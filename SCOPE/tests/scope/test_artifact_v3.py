"""LocalDecisionArtifactV3 schema tests."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, LocalDecisionArtifact, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import CapabilityId


def test_artifact_v3_schema_and_capability():
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
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": [], "remove_ids": ["d1"]},
        ),
        evidence_ids=("obs_1",),
        document_ids=("d1",),
        capability_id="duplicate_evidence",
        recommended_operation="skip_curate",
        teacher_source="EvidenceShadow",
        diagnosis="semantic_duplicate",
    )
    assert art.schema_version.startswith("scope.artifact")
    assert art.capability_id == "duplicate_evidence"
    assert art.resolved_capability() == CapabilityId.DUPLICATE_EVIDENCE
    assert art.recommended_operation == "skip_curate"
    assert isinstance(art, LocalDecisionArtifact)
    assert art.artifact_hash()


def test_artifact_v3_premature_stop_fields():
    student = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"answer": "maybe"},
    )
    art = PrivilegedArtifact.build(
        episode_id="ep1",
        turn_id=2,
        module_id="verification",
        mode=GuidanceMode.CORRECT,
        reason_code="PREMATURE_STOP",
        student_action=student,
        capability_id="premature_stop",
        recommended_operation="continue_search",
        operation_args={"query_intent": "fill_missing_claim"},
        runtime_fields_used=("remaining_turns",),
    )
    assert art.resolved_capability() == CapabilityId.PREMATURE_STOP
    assert "remaining_turns" in art.runtime_fields_used
