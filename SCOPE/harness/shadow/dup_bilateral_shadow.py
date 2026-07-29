"""Bilateral duplicate shadow: KEEP_EVIDENCE / SKIP_DUPLICATE only.

Triggered at evidence-admission decision points (student curate candidates),
not at error-triggered duplicate suspicion states.
"""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.validators import EvidenceVerifier, ValidationResult
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import CapabilityId
from harness.capability.dup_decision_point import (
    DECISION_TYPE_EVIDENCE_ADMISSION,
    DupDecisionPoint,
    build_decision_points,
    is_duplicate_candidate,
    is_evidence_admission_action,
)
from harness.capability.dup_operation import DupOperation
from harness.capability.state import DecisionState
from harness.shadow.base import ShadowModule


class DupBilateralShadow(ShadowModule):
    """Same-state shadow for bilateral KEEP/SKIP at curate decision points."""

    module_id = "duplicate_evidence"

    def __init__(self) -> None:
        self._verifier = EvidenceVerifier()

    def analyze_candidate(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
        decision_point: DupDecisionPoint,
    ) -> PrivilegedArtifact:
        """Shadow judgment for one curate candidate."""
        curated = decision_point.curated_evidence_ids
        cid = decision_point.candidate_evidence_id
        is_dup = is_duplicate_candidate(cid, curated)

        if is_dup:
            operation = DupOperation.SKIP_DUPLICATE
            mode = GuidanceMode.CORRECT
            reason = "DUPLICATE_EVIDENCE"
            recommended = CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={
                    "add_ids": [
                        d
                        for d in (student_action.arguments.get("add_ids") or [])
                        if str(d) != str(cid)
                    ],
                    "remove_ids": [],
                },
            )
        else:
            operation = DupOperation.KEEP_EVIDENCE
            mode = GuidanceMode.ENDORSE
            reason = "EVIDENCE_UPDATE_VALID"
            recommended = student_action

        return PrivilegedArtifact.build(
            episode_id=state.episode_id,
            turn_id=state.turn_id,
            module_id=self.module_id,
            mode=mode,
            reason_code=reason,
            student_action=student_action,
            recommended_action=recommended,
            evidence_ids=decision_point.observed_ids,
            document_ids=decision_point.curated_evidence_ids[:10],
            confidence=0.85 if is_dup else 0.7,
            metadata={
                "task_id": state.task_id,
                "schema_version": self.schema_version,
                "decision_type": DECISION_TYPE_EVIDENCE_ADMISSION,
                "decision_point": decision_point.to_dict(),
                "shadow_operation": operation.value,
                "candidate_is_duplicate": is_dup,
            },
            capability_id=CapabilityId.DUPLICATE_EVIDENCE.value,
            target=cid,
            diagnosis=reason.lower(),
            recommended_operation=operation.value,
            operation_args={"candidate_id": cid},
            teacher_source="DupBilateralShadow",
        )

    def analyze(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> PrivilegedArtifact:
        """Analyze first curate candidate (compat with ShadowModule protocol)."""
        points = build_decision_points(state, student_action)
        if not points:
            return PrivilegedArtifact.build(
                episode_id=state.episode_id,
                turn_id=state.turn_id,
                module_id=self.module_id,
                mode=GuidanceMode.IGNORE,
                reason_code="NOT_EVIDENCE_ADMISSION",
                student_action=student_action,
                capability_id=CapabilityId.DUPLICATE_EVIDENCE.value,
                teacher_source="DupBilateralShadow",
            )
        return self.analyze_candidate(state, student_action, points[0])

    def analyze_all_candidates(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> list[PrivilegedArtifact]:
        """Return one artifact per curate candidate (bilateral labeling)."""
        if not is_evidence_admission_action(student_action):
            return []
        return [
            self.analyze_candidate(state, student_action, dp)
            for dp in build_decision_points(state, student_action)
        ]

    def validate_candidate(
        self,
        state: DecisionState,
        candidate: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        return self._verifier.validate(state, candidate, artifact)
