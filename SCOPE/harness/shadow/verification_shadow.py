"""Verification typed shadow module."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.validators import ValidationResult, VerificationVerifier
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState
from harness.capability.stop_calibration import evidence_sufficient_for_stop
from harness.shadow.base import ShadowModule

# Actions that mean the student is about to (or has) stop / answer.
_STOPPING_ACTIONS = frozenset(
    {
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.ABSTAIN,
    }
)

_CONTINUING_ACTIONS = frozenset(
    {
        CapabilityActionType.SEARCH,
        CapabilityActionType.GREP,
        CapabilityActionType.OPEN_DOCUMENT,
        CapabilityActionType.REVIEW_DOCS,
        CapabilityActionType.CONTINUE_SEARCH,
        CapabilityActionType.REWRITE_QUERY,
    }
)


def _has_conflict(state: DecisionState) -> bool:
    for rec in state.verification_records:
        vals = list(rec.judgments.values())
        if vals and (True in vals) and (False in vals):
            return True
    return False


def _has_invalid_citation(state: DecisionState) -> bool:
    visible = set(state.visible_document_ids) | set(state.pool_document_ids)
    for rec in state.verification_records:
        for did in rec.document_ids:
            if did not in visible:
                return True
    for did in state.curated_document_ids:
        if did not in visible:
            return True
    return False


def _missing_direct_evidence(state: DecisionState) -> bool:
    """True when there is no usable direct support for answering."""
    curated = state.curated_document_ids
    if not curated:
        return True
    visible = set(state.visible_document_ids) | set(state.pool_document_ids)
    for claim in state.evidence_claims:
        if not claim.supporting_document_ids:
            return True
        if not any(d in visible for d in claim.supporting_document_ids):
            return True
    return False


def _unsupported_or_unverified(state: DecisionState) -> bool:
    """Hard unresolved / never-verified evidence."""
    if not state.verification_records:
        return True
    for rec in state.verification_records:
        vals = list(rec.judgments.values())
        if vals and not any(vals):
            return True
    return False


def _insufficient_evidence_for_stop(state: DecisionState) -> bool:
    """Core state: 证据不足 + 准备停止."""
    return (
        _missing_direct_evidence(state)
        or _unsupported_or_unverified(state)
        or _has_conflict(state)
    )


class VerificationShadow(ShadowModule):
    module_id = "verification"

    def __init__(self) -> None:
        self._verifier = VerificationVerifier()

    def analyze(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> PrivilegedArtifact:
        visible = set(state.visible_document_ids) | set(state.pool_document_ids)
        curated = set(state.curated_document_ids)
        evidence_ids = tuple(state.observation_ids[-5:])

        conflict = _has_conflict(state)
        invalid_citation = _has_invalid_citation(state)
        missing_evidence = _missing_direct_evidence(state)
        unsupported_stop = _unsupported_or_unverified(state)
        insufficient = _insufficient_evidence_for_stop(state)

        stopping = student_action.action_type in _STOPPING_ACTIONS
        continuing = student_action.action_type in _CONTINUING_ACTIONS
        verifying = student_action.action_type == CapabilityActionType.VERIFY_CLAIM

        recommended: CapabilityAction | None = None
        mode = GuidanceMode.IGNORE
        reason = "VERIFICATION_SUPPORTED"
        confidence = 0.3
        target_claim = student_action.target_claim_id

        if conflict and (stopping or verifying or curated):
            mode = GuidanceMode.CORRECT
            reason = "UNRESOLVED_CONFLICT"
            confidence = 0.85
            recommended = CapabilityAction(
                action_type=CapabilityActionType.SEARCH,
                arguments={"query": f"independent source for {state.query[:80]}"},
            )
        elif invalid_citation and (stopping or verifying):
            mode = GuidanceMode.CORRECT
            reason = "INVALID_CITATION"
            confidence = 0.8
            valid_doc = next(iter(visible), None)
            if valid_doc:
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.OPEN_DOCUMENT,
                    arguments={"doc_id": valid_doc},
                )
            else:
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.CONTINUE_SEARCH,
                    arguments={"query": state.query},
                )
        elif stopping and insufficient:
            # 证据不足 + 模型准备停止 / 直接作答 — must CORRECT
            mode = GuidanceMode.CORRECT
            if missing_evidence or not curated:
                reason = "MISSING_DIRECT_EVIDENCE"
            else:
                reason = "PREMATURE_STOP"
            confidence = 0.9
            if curated:
                claim_text = state.query
                if state.evidence_claims:
                    claim_text = state.evidence_claims[0].text
                    target_claim = state.evidence_claims[0].claim_id
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.VERIFY_CLAIM,
                    arguments={
                        "doc_ids": list(curated)[:5],
                        "claim": claim_text,
                    },
                    target_claim_id=target_claim,
                )
            else:
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.REWRITE_QUERY,
                    arguments={"query": state.query, "target_claim": state.query},
                    target_claim_id=target_claim,
                )
        elif stopping and not insufficient:
            # Stop with adequate verified support — endorse
            mode = GuidanceMode.ENDORSE
            reason = "VERIFICATION_SUPPORTED"
            confidence = 0.8
        elif continuing and evidence_sufficient_for_stop(state):
            # Bilateral stop calibration: student continues but evidence is sufficient.
            mode = GuidanceMode.CORRECT
            reason = "VERIFICATION_SUPPORTED"
            confidence = 0.82
            recommended = CapabilityAction(
                action_type=CapabilityActionType.STOP_AND_ANSWER,
                arguments={
                    "reasoning": "Evidence coverage sufficient; stop and answer.",
                },
            )
        elif verifying:
            docs = student_action.arguments.get("doc_ids") or []
            claim = str(student_action.arguments.get("claim", "")).strip()
            if not docs or not claim:
                mode = GuidanceMode.CORRECT
                reason = "MISSING_DIRECT_EVIDENCE"
                confidence = 0.85
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.VERIFY_CLAIM,
                    arguments={
                        "doc_ids": list(curated)[:5] or list(visible)[:3],
                        "claim": claim or state.query,
                    },
                    target_claim_id=target_claim,
                )
            elif all(str(d) in visible for d in docs):
                mode = GuidanceMode.ENDORSE
                reason = "VERIFICATION_SUPPORTED"
                confidence = 0.75
            else:
                mode = GuidanceMode.CORRECT
                reason = "SOURCE_NOT_VISIBLE"
                confidence = 0.8
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.OPEN_DOCUMENT,
                    arguments={"doc_id": next(iter(visible), "")},
                )
        else:
            # Non stop/answer/verify actions are not primary Verification targets.
            # Soft ignore (do not auto-endorse curate as "verified").
            mode = GuidanceMode.IGNORE
            reason = "VERIFICATION_SUPPORTED"
            confidence = 0.25

        from harness.capability.capability_id import CapabilityId, REASON_CODE_TO_CAPABILITY

        cap = REASON_CODE_TO_CAPABILITY.get(reason)
        # Stop decisions belong to premature_stop capability even when endorsed
        if stopping or (continuing and recommended is not None):
            cap = CapabilityId.PREMATURE_STOP
        runtime_fields: list[str] = []
        if cap == CapabilityId.PREMATURE_STOP:
            runtime_fields = ["remaining_turns"]
        op = ""
        op_args: dict = {}
        if recommended is not None:
            op = recommended.action_type.value
            op_args = dict(recommended.arguments)
            if cap == CapabilityId.PREMATURE_STOP and mode == GuidanceMode.CORRECT:
                if "query_intent" not in op_args:
                    op_args = {
                        **op_args,
                        "query_intent": "fill_missing_claim",
                    }
        elif stopping and mode == GuidanceMode.ENDORSE:
            op = "stop_and_answer"
            op_args = dict(student_action.arguments)

        return PrivilegedArtifact.build(
            episode_id=state.episode_id,
            turn_id=state.turn_id,
            module_id=self.module_id,
            mode=mode,
            reason_code=reason,
            student_action=student_action,
            recommended_action=recommended,
            target_claim_id=target_claim,
            evidence_ids=evidence_ids,
            document_ids=tuple(list(curated)[:10]),
            confidence=confidence,
            metadata={
                "task_id": state.task_id,
                "schema_version": self.schema_version,
                "insufficient_evidence": insufficient,
                "stopping": stopping,
                "continuing": continuing,
                "stop_calibration": continuing and recommended is not None,
                "unsupported_stop": unsupported_stop,
            },
            capability_id=cap.value if cap else "",
            diagnosis=reason.lower() if reason else "",
            recommended_operation=op,
            operation_args=op_args,
            runtime_fields_used=tuple(runtime_fields),
            teacher_source="VerificationShadow",
        )

    def validate_candidate(
        self,
        state: DecisionState,
        candidate: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        return self._verifier.validate(state, candidate, artifact)
