"""Evidence-state typed shadow module (deterministic heuristics)."""

from __future__ import annotations

import re

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.validators import EvidenceVerifier, ValidationResult
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState
from harness.shadow.base import ShadowModule

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")


def _query_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _doc_context_overlap(rendered: str, doc_id: str, query_toks: set[str]) -> float:
    """Cheap relevance: token overlap between query and text near a doc id mention."""
    if not rendered or not doc_id or not query_toks:
        return 0.0
    idx = rendered.find(str(doc_id))
    if idx < 0:
        return 0.0
    window = rendered[max(0, idx - 120) : idx + len(str(doc_id)) + 240]
    doc_toks = _query_tokens(window)
    if not doc_toks:
        return 0.0
    return len(query_toks & doc_toks) / max(1, len(query_toks))


def _conflicting_evidence(state: DecisionState) -> bool:
    for claim in state.evidence_claims:
        if str(claim.status).lower() in {"conflict", "conflicting", "contradicted"}:
            return True
    for rec in state.verification_records:
        vals = list(rec.judgments.values())
        if vals and (True in vals) and (False in vals):
            return True
    return False


def _wrong_claim_binding(state: DecisionState) -> bool:
    pool = set(state.pool_document_ids) | set(state.visible_document_ids)
    for claim in state.evidence_claims:
        supports = claim.supporting_document_ids
        if not supports:
            continue
        # Bound docs are outside the visible/pool set
        if not any(d in pool for d in supports):
            return True
        # Bound docs never curated and claim marked supported/verified
        if claim.status.lower() in {"supported", "verified", "linked"}:
            curated = set(state.curated_document_ids)
            if supports and not any(d in curated for d in supports):
                return True
    return False


def _missing_direct_support(state: DecisionState) -> bool:
    if not state.evidence_claims:
        return False
    return any(not c.supporting_document_ids for c in state.evidence_claims)


def _weak_support(state: DecisionState, student_action: CapabilityAction) -> bool:
    weak_statuses = {"unsupported", "weak", "unverified", "partial", "unknown"}
    for claim in state.evidence_claims:
        if claim.supporting_document_ids and claim.status.lower() in weak_statuses:
            return True
        # Single thin support without any positive verification
        if len(claim.supporting_document_ids) == 1 and not any(
            any(r.judgments.values()) for r in state.verification_records if r.judgments
        ):
            return True
    status = str(student_action.arguments.get("status", "")).lower()
    if status in {"supported", "verified"} and not state.verification_records:
        return True
    return False


def _irrelevant_evidence(
    state: DecisionState,
    add_ids: list,
) -> bool:
    query_toks = _query_tokens(state.query)
    if not query_toks:
        return False
    rendered = state.rendered_context or ""
    candidates = [str(d) for d in add_ids] if add_ids else list(state.curated_document_ids)[-3:]
    if not candidates:
        return False
    # Irrelevant if every candidate has near-zero overlap with the query in context.
    # Require a non-trivial context window (avoid false positives on id-only lines).
    overlaps = [_doc_context_overlap(rendered, d, query_toks) for d in candidates]
    windows_ok = []
    for d in candidates:
        idx = rendered.find(str(d))
        if idx < 0:
            windows_ok.append(False)
            continue
        window = rendered[max(0, idx - 120) : idx + len(str(d)) + 240]
        windows_ok.append(len(window.strip()) >= 40)
    if (
        overlaps
        and windows_ok
        and all(windows_ok)
        and all(o < 0.05 for o in overlaps)
        and any(str(d) in rendered for d in candidates)
    ):
        return True
    # Or: claims exist with bindings, but curated/added docs are outside those bindings
    if state.evidence_claims and add_ids:
        bound = {
            d
            for c in state.evidence_claims
            for d in c.supporting_document_ids
        }
        if bound and all(str(d) not in bound for d in add_ids):
            return True
    return False


class EvidenceShadow(ShadowModule):
    module_id = "evidence_state"

    def __init__(self) -> None:
        self._verifier = EvidenceVerifier()

    def analyze(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> PrivilegedArtifact:
        curated = list(state.curated_document_ids)
        pool = set(state.pool_document_ids)
        evidence_ids = tuple(state.observation_ids[-5:])

        add_ids = student_action.arguments.get("add_ids") or []
        if not isinstance(add_ids, list):
            add_ids = []
        duplicate = bool(add_ids) and all(str(d) in curated for d in add_ids)

        # Status consistency with verification records
        invalid_status = False
        status = str(student_action.arguments.get("status", "")).lower()
        if status in {"verified", "supported"} and not state.verification_records:
            invalid_status = True

        mode = GuidanceMode.IGNORE
        reason = "EVIDENCE_UPDATE_VALID"
        confidence = 0.4
        recommended: CapabilityAction | None = None
        target_claim = student_action.target_claim_id

        conflicting = _conflicting_evidence(state)
        wrong_binding = _wrong_claim_binding(state)
        missing_support = _missing_direct_support(state)
        weak = _weak_support(state, student_action)
        irrelevant = _irrelevant_evidence(state, add_ids)

        if invalid_status:
            mode = GuidanceMode.CORRECT
            reason = "INVALID_STATUS_TRANSITION"
            confidence = 0.85
            recommended = CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={"status": "unsupported", "add_ids": [], "remove_ids": []},
                target_claim_id=target_claim,
            )
        elif duplicate:
            mode = GuidanceMode.CORRECT
            reason = "DUPLICATE_EVIDENCE"
            confidence = 0.8
            candidates = [d for d in state.pool_document_ids if d not in curated]
            recommended = CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={
                    "add_ids": candidates[:1],
                    "remove_ids": list(add_ids)[:1] if add_ids else [],
                },
            )
        elif conflicting:
            mode = GuidanceMode.CORRECT
            reason = "CONFLICTING_EVIDENCE"
            confidence = 0.85
            claim = next(
                (
                    c
                    for c in state.evidence_claims
                    if str(c.status).lower() in {"conflict", "conflicting", "contradicted"}
                ),
                state.evidence_claims[0] if state.evidence_claims else None,
            )
            target_claim = claim.claim_id if claim else target_claim
            recommended = CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={"status": "conflict", "add_ids": [], "remove_ids": []},
                target_claim_id=target_claim,
            )
        elif wrong_binding:
            mode = GuidanceMode.CORRECT
            reason = "WRONG_CLAIM_BINDING"
            confidence = 0.8
            claim = next(
                (
                    c
                    for c in state.evidence_claims
                    if c.supporting_document_ids
                    and not any(
                        d in pool or d in set(state.visible_document_ids)
                        for d in c.supporting_document_ids
                    )
                ),
                state.evidence_claims[0] if state.evidence_claims else None,
            )
            bind_doc = curated[0] if curated else (next(iter(pool), None))
            target_claim = claim.claim_id if claim else target_claim
            recommended = CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={
                    "add_ids": [bind_doc] if bind_doc else [],
                    "status": "linked",
                },
                target_claim_id=target_claim,
            )
        elif missing_support:
            mode = GuidanceMode.CORRECT
            reason = "MISSING_DIRECT_SUPPORT"
            confidence = 0.8
            claim = next(
                (c for c in state.evidence_claims if not c.supporting_document_ids),
                None,
            )
            target_claim = claim.claim_id if claim else target_claim
            bind_doc = curated[0] if curated else None
            recommended = CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={
                    "status": "unsupported",
                    "add_ids": [bind_doc] if bind_doc else [],
                },
                target_claim_id=target_claim,
            )
        elif irrelevant and student_action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
            CapabilityActionType.REVIEW_DOCS,
        }:
            mode = GuidanceMode.CORRECT
            reason = "IRRELEVANT_EVIDENCE"
            confidence = 0.75
            recommended = CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={
                    "add_ids": [],
                    "remove_ids": list(add_ids)[:3] if add_ids else list(curated)[:1],
                },
            )
        elif weak and student_action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
            CapabilityActionType.VERIFY_CLAIM,
            CapabilityActionType.REVIEW_DOCS,
        }:
            mode = GuidanceMode.CORRECT
            reason = "WEAK_SUPPORT"
            confidence = 0.75
            claim = state.evidence_claims[0] if state.evidence_claims else None
            target_claim = claim.claim_id if claim else target_claim
            recommended = CapabilityAction(
                action_type=CapabilityActionType.UPDATE_EVIDENCE,
                arguments={"status": "unsupported"},
                target_claim_id=target_claim,
            )
        elif student_action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
            CapabilityActionType.REVIEW_DOCS,
        }:
            bad_add = [d for d in add_ids if str(d) not in pool]
            if bad_add:
                mode = GuidanceMode.CORRECT
                reason = "WEAK_SOURCE_ONLY"
                confidence = 0.7
                recommended = CapabilityAction(
                    action_type=CapabilityActionType.CURATE_DOCUMENT,
                    arguments={
                        "add_ids": [d for d in curated[:1]],
                        "remove_ids": bad_add,
                    },
                )
            else:
                mode = GuidanceMode.ENDORSE
                reason = "EVIDENCE_UPDATE_VALID"
                confidence = 0.7

        from harness.capability.capability_id import REASON_CODE_TO_CAPABILITY

        cap = REASON_CODE_TO_CAPABILITY.get(reason)
        op = ""
        op_args: dict = {}
        target = None
        if recommended is not None:
            op = recommended.action_type.value
            op_args = dict(recommended.arguments)
        if reason == "DUPLICATE_EVIDENCE":
            op = op or "skip_curate"
            target = str(add_ids[0]) if add_ids else None

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
            document_ids=tuple(curated[:10]),
            confidence=confidence,
            metadata={"task_id": state.task_id, "schema_version": self.schema_version},
            capability_id=cap.value if cap else "",
            target=target,
            diagnosis=reason.lower() if reason else "",
            recommended_operation=op,
            operation_args=op_args,
            teacher_source="EvidenceShadow",
        )

    def validate_candidate(
        self,
        state: DecisionState,
        candidate: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        return self._verifier.validate(state, candidate, artifact)
