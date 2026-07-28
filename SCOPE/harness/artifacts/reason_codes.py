"""Closed-set reason codes for SCOPE artifacts (legacy + V3 capability codes).

Training must use closed-set codes only. Free text goes in debug_reason
(loss_mask = 0).
"""

from __future__ import annotations

from harness.capability.capability_id import CapabilityId

# --- Legacy module-level reason codes (audit / shadow BC) ---

VERIFICATION_REASON_CODES = frozenset(
    {
        "VERIFICATION_SUPPORTED",
        "MISSING_DIRECT_EVIDENCE",
        "UNRESOLVED_CONFLICT",
        "INVALID_CITATION",
        "SOURCE_NOT_VISIBLE",
        "PREMATURE_STOP",
    }
)

EVIDENCE_REASON_CODES = frozenset(
    {
        "EVIDENCE_UPDATE_VALID",
        "CLAIM_WITHOUT_SUPPORT",
        "DUPLICATE_EVIDENCE",
        "IRRELEVANT_EVIDENCE",
        "WEAK_SUPPORT",
        "MISSING_DIRECT_SUPPORT",
        "CONFLICTING_EVIDENCE",
        "WRONG_CLAIM_BINDING",
        "WEAK_SOURCE_ONLY",
        "MISSING_CLAIM_LINK",
        "INVALID_STATUS_TRANSITION",
    }
)

BUDGET_REASON_CODES = frozenset(
    {
        "REPEATED_QUERY",
        "LOW_INFORMATION_GAIN",
        "COVERAGE_SUFFICIENT",
        "BUDGET_EXHAUSTION_RISK",
    }
)

ALL_REASON_CODES = (
    VERIFICATION_REASON_CODES | EVIDENCE_REASON_CODES | BUDGET_REASON_CODES
)

# --- V3 capability-scoped fine-grained codes ---

DUPLICATE_EVIDENCE_REASON_CODES = frozenset(
    {
        "exact_duplicate",
        "normalized_url_duplicate",
        "semantic_duplicate",
        # legacy uppercase aliases accepted
        "DUPLICATE_EVIDENCE",
    }
)

PREMATURE_STOP_REASON_CODES = frozenset(
    {
        "unsupported_claim_remaining",
        "insufficient_coverage",
        "unresolved_conflict",
        "answer_not_grounded",
        "coverage_insufficient",
        # legacy
        "PREMATURE_STOP",
        "MISSING_DIRECT_EVIDENCE",
        "UNRESOLVED_CONFLICT",
    }
)

CAPABILITY_REASON_CODES: dict[CapabilityId, frozenset[str]] = {
    CapabilityId.DUPLICATE_EVIDENCE: DUPLICATE_EVIDENCE_REASON_CODES,
    CapabilityId.PREMATURE_STOP: PREMATURE_STOP_REASON_CODES,
    CapabilityId.IRRELEVANT_EVIDENCE: frozenset({"IRRELEVANT_EVIDENCE", "irrelevant_evidence"}),
    CapabilityId.INVALID_CITATION: frozenset(
        {"INVALID_CITATION", "SOURCE_NOT_VISIBLE", "invalid_citation"}
    ),
}

# Legacy uppercase → preferred V3 snake_case
LEGACY_TO_V3_REASON: dict[str, str] = {
    "DUPLICATE_EVIDENCE": "semantic_duplicate",
    "PREMATURE_STOP": "insufficient_coverage",
    "MISSING_DIRECT_EVIDENCE": "unsupported_claim_remaining",
    "UNRESOLVED_CONFLICT": "unresolved_conflict",
    "IRRELEVANT_EVIDENCE": "irrelevant_evidence",
    "INVALID_CITATION": "invalid_citation",
}


def is_valid_reason_code(code: str) -> bool:
    if code in ALL_REASON_CODES:
        return True
    for codes in CAPABILITY_REASON_CODES.values():
        if code in codes:
            return True
    return False


def normalize_reason_code(code: str, capability: CapabilityId | None = None) -> str:
    raw = str(code or "").strip()
    if not raw:
        return raw
    if capability is not None:
        allowed = CAPABILITY_REASON_CODES.get(capability)
        if allowed and raw in allowed:
            return LEGACY_TO_V3_REASON.get(raw, raw)
        if allowed and raw.lower() in {c.lower() for c in allowed}:
            for c in allowed:
                if c.lower() == raw.lower():
                    return LEGACY_TO_V3_REASON.get(c, c)
    return LEGACY_TO_V3_REASON.get(raw, raw)


def is_valid_for_capability(code: str, capability: CapabilityId) -> bool:
    allowed = CAPABILITY_REASON_CODES.get(capability)
    if not allowed:
        return is_valid_reason_code(code)
    return code in allowed or LEGACY_TO_V3_REASON.get(code, code) in allowed
