"""Rule-based (and pluggable) candidate action generators for Correct mode."""

from __future__ import annotations

from typing import Protocol

from harness.artifacts.schema import PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState


class CandidateGenerator(Protocol):
    def generate(
        self,
        state: DecisionState,
        artifact: PrivilegedArtifact,
    ) -> list[CapabilityAction]:
        ...


class RuleBasedCandidateGenerator:
    """Deterministic templates keyed by reason_code (SCOPE §14.1)."""

    def generate(
        self,
        state: DecisionState,
        artifact: PrivilegedArtifact,
    ) -> list[CapabilityAction]:
        code = artifact.reason_code
        curated = list(state.curated_document_ids)
        pool = list(state.pool_document_ids)
        claim = artifact.target_claim_id
        claim_text = state.query
        for c in state.evidence_claims:
            if claim and c.claim_id == claim:
                claim_text = c.text
                break

        mapping: dict[str, CapabilityAction] = {
            "PREMATURE_STOP": CapabilityAction(
                action_type=CapabilityActionType.VERIFY_CLAIM,
                arguments={"doc_ids": curated[:5] or pool[:3], "claim": claim_text},
                target_claim_id=claim,
            ),
            "MISSING_DIRECT_EVIDENCE": CapabilityAction(
                action_type=CapabilityActionType.REWRITE_QUERY,
                arguments={"query": claim_text, "target_claim": claim_text},
                target_claim_id=claim,
            ),
            "UNRESOLVED_CONFLICT": CapabilityAction(
                action_type=CapabilityActionType.SEARCH,
                arguments={"query": f"independent source {claim_text[:80]}"},
                target_claim_id=claim,
            ),
            "INVALID_CITATION": CapabilityAction(
                action_type=CapabilityActionType.OPEN_DOCUMENT,
                arguments={"doc_id": (pool[0] if pool else (curated[0] if curated else ""))},
            ),
            "SOURCE_NOT_VISIBLE": CapabilityAction(
                action_type=CapabilityActionType.OPEN_DOCUMENT,
                arguments={"doc_id": pool[0] if pool else ""},
            ),
            "CLAIM_WITHOUT_SUPPORT": CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={"status": "unsupported"},
                target_claim_id=claim,
            ),
            "MISSING_DIRECT_SUPPORT": CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={
                    "status": "unsupported",
                    "add_ids": curated[:1] or pool[:1],
                },
                target_claim_id=claim,
            ),
            "DUPLICATE_EVIDENCE": CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={
                    "add_ids": [d for d in pool if d not in curated][:1],
                    "remove_ids": [],
                },
            ),
            "IRRELEVANT_EVIDENCE": CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={
                    "add_ids": [],
                    "remove_ids": curated[:1],
                },
            ),
            "WEAK_SUPPORT": CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={"status": "unsupported"},
                target_claim_id=claim,
            ),
            "CONFLICTING_EVIDENCE": CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={"status": "conflict"},
                target_claim_id=claim,
            ),
            "WRONG_CLAIM_BINDING": CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={
                    "add_ids": curated[:1] or pool[:1],
                    "status": "linked",
                },
                target_claim_id=claim,
            ),
            "MISSING_CLAIM_LINK": CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={
                    "add_ids": curated[:1] or pool[:1],
                    "status": "linked",
                },
                target_claim_id=claim,
            ),
            "REPEATED_QUERY": CapabilityAction(
                action_type=CapabilityActionType.REWRITE_QUERY,
                arguments={"query": f"{state.query} alternative"},
            ),
            "LOW_INFORMATION_GAIN": CapabilityAction(
                action_type=CapabilityActionType.REVIEW_DOCS,
                arguments={"doc_ids": pool[:3]},
            ),
            "BUDGET_EXHAUSTION_RISK": CapabilityAction(
                action_type=CapabilityActionType.STOP_AND_ANSWER,
                arguments={"reasoning": "budget risk"},
            ),
        }
        action = mapping.get(code)
        if action is None and artifact.recommended_action is not None:
            return [artifact.recommended_action]
        return [action] if action is not None else []


def fill_recommended_action(
    state: DecisionState,
    artifact: PrivilegedArtifact,
    generator: CandidateGenerator | None = None,
) -> PrivilegedArtifact:
    """Ensure artifact has a recommended_action for CORRECT mode."""
    if artifact.recommended_action is not None:
        return artifact
    gen = generator or RuleBasedCandidateGenerator()
    cands = gen.generate(state, artifact)
    if not cands:
        return artifact
    from dataclasses import replace

    return replace(artifact, recommended_action=cands[0])
