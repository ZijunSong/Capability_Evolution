"""Local verifiers for recommended / student capability actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harness.artifacts.schema import PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "reasons": list(self.reasons),
        }


class LocalVerifier(Protocol):
    def validate(
        self,
        state: DecisionState,
        action: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        ...


class VerificationVerifier:
    def validate(
        self,
        state: DecisionState,
        action: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        reasons: list[str] = []
        visible_docs = set(state.visible_document_ids) | set(state.pool_document_ids)
        claim_ids = {c.claim_id for c in state.evidence_claims}

        if action.target_claim_id and action.target_claim_id not in claim_ids:
            # Allow claim text hash style ids not yet registered when correcting stop
            if action.action_type not in {
                CapabilityActionType.VERIFY_CLAIM,
                CapabilityActionType.CONTINUE_SEARCH,
                CapabilityActionType.REWRITE_QUERY,
            }:
                reasons.append("claim_id_missing")

        if action.action_type == CapabilityActionType.VERIFY_CLAIM:
            doc_ids = action.arguments.get("doc_ids") or []
            if not isinstance(doc_ids, list) or not doc_ids:
                reasons.append("verify_missing_doc_ids")
            else:
                for d in doc_ids:
                    if str(d) not in visible_docs:
                        reasons.append(f"cited_doc_not_visible:{d}")
            if not str(action.arguments.get("claim", "")).strip():
                reasons.append("verify_missing_claim")

        if action.action_type == CapabilityActionType.STOP_AND_ANSWER or (
            action.action_type == CapabilityActionType.ANSWER
        ):
            hard = [
                r
                for r in state.verification_records
                if r.judgments and not any(r.judgments.values())
            ]
            if hard and artifact.reason_code == "PREMATURE_STOP":
                # Stopping despite hard unresolved is invalid as endorsement target
                reasons.append("hard_unresolved_on_stop")

        # Consistency with reason code
        expected = {
            "PREMATURE_STOP": {
                CapabilityActionType.VERIFY_CLAIM,
                CapabilityActionType.CONTINUE_SEARCH,
            },
            "MISSING_DIRECT_EVIDENCE": {
                CapabilityActionType.REWRITE_QUERY,
                CapabilityActionType.SEARCH,
                CapabilityActionType.CONTINUE_SEARCH,
                CapabilityActionType.VERIFY_CLAIM,
            },
            "UNRESOLVED_CONFLICT": {
                CapabilityActionType.SEARCH,
                CapabilityActionType.VERIFY_CLAIM,
            },
            "INVALID_CITATION": {
                CapabilityActionType.OPEN_DOCUMENT,
                CapabilityActionType.CURATE_DOCUMENT,
            },
            "SOURCE_NOT_VISIBLE": {
                CapabilityActionType.OPEN_DOCUMENT,
                CapabilityActionType.CURATE_DOCUMENT,
            },
        }
        allowed = expected.get(artifact.reason_code)
        if allowed and action.action_type not in allowed and artifact.mode.value == "correct":
            reasons.append("action_reason_mismatch")

        score = 1.0 if not reasons else max(0.0, 1.0 - 0.25 * len(reasons))
        return ValidationResult(valid=not reasons, score=score, reasons=tuple(reasons))


class EvidenceVerifier:
    def validate(
        self,
        state: DecisionState,
        action: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        reasons: list[str] = []
        visible_docs = set(state.visible_document_ids) | set(state.pool_document_ids)

        if action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
        }:
            add_ids = action.arguments.get("add_ids") or []
            for d in add_ids if isinstance(add_ids, list) else []:
                if str(d) not in visible_docs:
                    reasons.append(f"doc_not_in_pool:{d}")
            status = str(action.arguments.get("status", "")).lower()
            if status == "verified":
                # Cannot mark verified without verification record
                claim = action.target_claim_id
                matched = [
                    r
                    for r in state.verification_records
                    if claim and claim in r.claim
                ]
                if not matched and not any(any(r.judgments.values()) for r in state.verification_records):
                    reasons.append("unverified_marked_verified")

        if artifact.reason_code == "DUPLICATE_EVIDENCE":
            if action.action_type != CapabilityActionType.CURATE_DOCUMENT:
                reasons.append("duplicate_requires_curate")

        if artifact.reason_code in {
            "MISSING_CLAIM_LINK",
            "WRONG_CLAIM_BINDING",
            "MISSING_DIRECT_SUPPORT",
            "CLAIM_WITHOUT_SUPPORT",
            "WEAK_SUPPORT",
            "CONFLICTING_EVIDENCE",
        }:
            if action.action_type not in {
                CapabilityActionType.UPDATE_EVIDENCE,
                CapabilityActionType.CURATE_DOCUMENT,
                CapabilityActionType.VERIFY_CLAIM,
            }:
                reasons.append("evidence_fix_requires_update")

        if artifact.reason_code == "IRRELEVANT_EVIDENCE":
            if action.action_type != CapabilityActionType.CURATE_DOCUMENT:
                reasons.append("irrelevant_requires_curate")

        score = 1.0 if not reasons else max(0.0, 1.0 - 0.25 * len(reasons))
        return ValidationResult(valid=not reasons, score=score, reasons=tuple(reasons))


class BudgetVerifier:
    def validate(
        self,
        state: DecisionState,
        action: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        reasons: list[str] = []
        if artifact.reason_code == "REPEATED_QUERY":
            if action.action_type not in {
                CapabilityActionType.REWRITE_QUERY,
                CapabilityActionType.STOP_AND_ANSWER,
                CapabilityActionType.OPEN_DOCUMENT,
            }:
                reasons.append("repeated_query_bad_fix")
        if artifact.reason_code == "BUDGET_EXHAUSTION_RISK":
            if state.remaining_turns > 5 and action.action_type == CapabilityActionType.STOP_AND_ANSWER:
                reasons.append("stop_not_justified_by_budget")
        score = 1.0 if not reasons else max(0.0, 1.0 - 0.3 * len(reasons))
        return ValidationResult(valid=not reasons, score=score, reasons=tuple(reasons))


def get_verifier(module_id: str) -> LocalVerifier:
    if module_id == "verification":
        return VerificationVerifier()
    if module_id == "evidence_state":
        return EvidenceVerifier()
    if module_id == "budget_control":
        return BudgetVerifier()
    return VerificationVerifier()
