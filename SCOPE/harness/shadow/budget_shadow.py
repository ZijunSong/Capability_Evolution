"""Budget-control typed shadow module."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.validators import BudgetVerifier, ValidationResult
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState
from harness.shadow.base import ShadowModule


class BudgetShadow(ShadowModule):
    module_id = "budget_control"

    def __init__(self) -> None:
        self._verifier = BudgetVerifier()

    def analyze(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> PrivilegedArtifact:
        evidence_ids = tuple(state.observation_ids[-3:])
        repeated = float(state.repeated_query_score or 0.0)
        low_turns = state.remaining_turns <= 3
        coverage_ok = len(state.curated_document_ids) >= 3

        mode = GuidanceMode.IGNORE
        reason = "COVERAGE_SUFFICIENT"
        confidence = 0.4
        recommended: CapabilityAction | None = None

        if repeated >= 0.8 and student_action.action_type in {
            CapabilityActionType.SEARCH,
            CapabilityActionType.REWRITE_QUERY,
            CapabilityActionType.CONTINUE_SEARCH,
        }:
            mode = GuidanceMode.CORRECT
            reason = "REPEATED_QUERY"
            confidence = 0.85
            recommended = CapabilityAction(
                action_type=CapabilityActionType.REWRITE_QUERY,
                arguments={"query": f"{state.query} alternative angle"},
            )
        elif low_turns and student_action.action_type == CapabilityActionType.SEARCH:
            mode = GuidanceMode.CORRECT
            reason = "BUDGET_EXHAUSTION_RISK"
            confidence = 0.8
            if coverage_ok:
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.STOP_AND_ANSWER,
                    arguments={"reasoning": "budget low; coverage sufficient"},
                )
            else:
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.OPEN_DOCUMENT,
                    arguments={
                        "doc_id": next(iter(state.pool_document_ids), ""),
                    },
                )
        elif coverage_ok and student_action.action_type in {
            CapabilityActionType.STOP_AND_ANSWER,
            CapabilityActionType.ANSWER,
        }:
            mode = GuidanceMode.ENDORSE
            reason = "COVERAGE_SUFFICIENT"
            confidence = 0.7
        elif (
            len(state.action_history) >= 4
            and student_action.action_type == CapabilityActionType.SEARCH
            and len(state.curated_document_ids) == 0
        ):
            mode = GuidanceMode.CORRECT
            reason = "LOW_INFORMATION_GAIN"
            confidence = 0.75
            recommended = CapabilityAction(
                action_type=CapabilityActionType.REVIEW_DOCS,
                arguments={"doc_ids": list(state.pool_document_ids)[:3]},
            )

        from harness.capability.capability_id import REASON_CODE_TO_CAPABILITY

        cap = REASON_CODE_TO_CAPABILITY.get(reason)
        runtime_fields: list[str] = []
        if reason in {"BUDGET_EXHAUSTION_RISK", "REPEATED_QUERY"}:
            runtime_fields = ["remaining_turns", "repeated_query_score"]
        op = recommended.action_type.value if recommended else ""
        op_args = dict(recommended.arguments) if recommended else {}

        return PrivilegedArtifact.build(
            episode_id=state.episode_id,
            turn_id=state.turn_id,
            module_id=self.module_id,
            mode=mode,
            reason_code=reason,
            student_action=student_action,
            recommended_action=recommended,
            target_claim_id=student_action.target_claim_id,
            evidence_ids=evidence_ids,
            document_ids=tuple(state.curated_document_ids[:5]),
            confidence=confidence,
            metadata={"task_id": state.task_id, "schema_version": self.schema_version},
            capability_id=cap.value if cap else "",
            diagnosis=reason.lower() if reason else "",
            recommended_operation=op,
            operation_args=op_args,
            runtime_fields_used=tuple(runtime_fields),
            teacher_source="BudgetShadow",
        )

    def validate_candidate(
        self,
        state: DecisionState,
        candidate: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        return self._verifier.validate(state, candidate, artifact)
