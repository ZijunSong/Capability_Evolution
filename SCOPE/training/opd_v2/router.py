"""Endorse / Correct / Ignore guidance router."""

from __future__ import annotations

from dataclasses import dataclass

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.validators import ValidationResult, get_verifier
from harness.artifacts.visibility import check_artifact_visibility, mask_artifact_if_invalid
from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState
from harness.shadow.base import ShadowModule


@dataclass(frozen=True)
class GuidanceDecision:
    mode: GuidanceMode
    artifact: PrivilegedArtifact
    validation: ValidationResult


class GuidanceRouter:
    """Route artifact + local verifier into ENDORSE / CORRECT / IGNORE.

    Endorse is based on local module agreement, NOT final episode reward.
    """

    def route(
        self,
        state: DecisionState,
        artifact: PrivilegedArtifact,
        *,
        module: ShadowModule | None = None,
    ) -> GuidanceDecision:
        artifact, vis = mask_artifact_if_invalid(state, artifact)
        if not vis.valid or artifact.mode == GuidanceMode.IGNORE:
            return GuidanceDecision(
                mode=GuidanceMode.IGNORE,
                artifact=artifact,
                validation=ValidationResult(
                    valid=False,
                    score=0.0,
                    reasons=vis.violations or ("ignored",),
                ),
            )

        verifier = get_verifier(artifact.module_id)
        student_val = verifier.validate(state, artifact.student_action, artifact)

        if artifact.mode == GuidanceMode.ENDORSE and student_val.valid:
            return GuidanceDecision(
                mode=GuidanceMode.ENDORSE,
                artifact=artifact,
                validation=student_val,
            )

        if artifact.mode == GuidanceMode.CORRECT and artifact.recommended_action is not None:
            if module is not None:
                cand_val = module.validate_candidate(
                    state, artifact.recommended_action, artifact
                )
            else:
                cand_val = verifier.validate(state, artifact.recommended_action, artifact)
            # Also require visibility of recommended action
            vis2 = check_artifact_visibility(state, artifact)
            if cand_val.valid and vis2.valid:
                return GuidanceDecision(
                    mode=GuidanceMode.CORRECT,
                    artifact=artifact,
                    validation=cand_val,
                )
            return GuidanceDecision(
                mode=GuidanceMode.IGNORE,
                artifact=artifact,
                validation=ValidationResult(
                    valid=False,
                    score=cand_val.score,
                    reasons=cand_val.reasons + (("visibility_failed",) if not vis2.valid else ()),
                ),
            )

        return GuidanceDecision(
            mode=GuidanceMode.IGNORE,
            artifact=artifact,
            validation=ValidationResult(
                valid=False,
                score=student_val.score,
                reasons=student_val.reasons or ("no_route",),
            ),
        )
