"""Capability IDs for SCOPE supervision (module-independent analysis units).

Paper-1 Round 1 locks:
  ENABLED:  duplicate_evidence, premature_stop
  DISABLED: irrelevant_evidence, invalid_citation
"""

from __future__ import annotations

from enum import Enum


class CapabilityId(str, Enum):
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    PREMATURE_STOP = "premature_stop"
    STOP_DECISION = "stop_decision"  # E0 alias for premature_stop
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"
    INVALID_CITATION = "invalid_citation"
    # E0 distillability probes
    EVIDENCE_CURATION = "evidence_curation"
    VERIFICATION_DECISION = "verification_decision"
    EXTERNAL_VERIFICATION = "external_verification"
    DETERMINISTIC_TRUNCATION = "deterministic_truncation"
    # Future / audit-only
    EVIDENCE_PRIORITIZATION = "evidence_prioritization"
    SUBTRACTIVE_CURATION = "subtractive_curation"
    MISSING_PRIMARY_SOURCE = "missing_primary_source"
    REPEATED_QUERY = "repeated_query"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    UNKNOWN = "unknown"


# Round-1 training allowlist
ROUND1_ENABLED_CAPABILITIES: frozenset[CapabilityId] = frozenset(
    {
        CapabilityId.DUPLICATE_EVIDENCE,
        CapabilityId.PREMATURE_STOP,
    }
)

ROUND1_DISABLED_CAPABILITIES: frozenset[CapabilityId] = frozenset(
    {
        CapabilityId.IRRELEVANT_EVIDENCE,
        CapabilityId.INVALID_CITATION,
    }
)

# Default module ownership (capability ≠ module)
CAPABILITY_DEFAULT_MODULE: dict[CapabilityId, str] = {
    CapabilityId.DUPLICATE_EVIDENCE: "evidence_state",
    CapabilityId.IRRELEVANT_EVIDENCE: "evidence_state",
    CapabilityId.EVIDENCE_PRIORITIZATION: "evidence_state",
    CapabilityId.SUBTRACTIVE_CURATION: "evidence_state",
    CapabilityId.EVIDENCE_CURATION: "evidence_state",
    CapabilityId.MISSING_PRIMARY_SOURCE: "evidence_state",
    CapabilityId.PREMATURE_STOP: "verification",
    CapabilityId.STOP_DECISION: "verification",
    CapabilityId.VERIFICATION_DECISION: "verification",
    CapabilityId.EXTERNAL_VERIFICATION: "verification",
    CapabilityId.INVALID_CITATION: "verification",
    CapabilityId.DETERMINISTIC_TRUNCATION: "context_budget",
    CapabilityId.REPEATED_QUERY: "budget_control",
    CapabilityId.BUDGET_EXHAUSTION: "budget_control",
}

# E0 distillability pilot capabilities (capability-level, not module-level)
E0_PROBE_CAPABILITIES: tuple[CapabilityId, ...] = (
    CapabilityId.DUPLICATE_EVIDENCE,
    CapabilityId.STOP_DECISION,
    CapabilityId.EVIDENCE_CURATION,
    CapabilityId.VERIFICATION_DECISION,
    CapabilityId.EXTERNAL_VERIFICATION,
    CapabilityId.DETERMINISTIC_TRUNCATION,
)

# Map E0 probe names to shadow/audit capability ids
E0_CAPABILITY_ALIASES: dict[CapabilityId, CapabilityId] = {
    CapabilityId.STOP_DECISION: CapabilityId.PREMATURE_STOP,
}

# Map legacy closed-set reason codes → capability
REASON_CODE_TO_CAPABILITY: dict[str, CapabilityId] = {
    "DUPLICATE_EVIDENCE": CapabilityId.DUPLICATE_EVIDENCE,
    "PREMATURE_STOP": CapabilityId.PREMATURE_STOP,
    "IRRELEVANT_EVIDENCE": CapabilityId.IRRELEVANT_EVIDENCE,
    "INVALID_CITATION": CapabilityId.INVALID_CITATION,
    "SOURCE_NOT_VISIBLE": CapabilityId.INVALID_CITATION,
    "REPEATED_QUERY": CapabilityId.REPEATED_QUERY,
    "BUDGET_EXHAUSTION_RISK": CapabilityId.BUDGET_EXHAUSTION,
}


def resolve_e0_capability(value: str | CapabilityId | None) -> CapabilityId:
    """Resolve E0 probe id to canonical shadow capability id."""
    cap = parse_capability_id(value)
    return E0_CAPABILITY_ALIASES.get(cap, cap)


def parse_capability_id(value: str | CapabilityId | None) -> CapabilityId:
    if value is None:
        return CapabilityId.UNKNOWN
    if isinstance(value, CapabilityId):
        return value
    raw = str(value).strip()
    if raw == "stop_decision":
        return CapabilityId.STOP_DECISION
    try:
        return CapabilityId(raw)
    except ValueError:
        pass
    upper = raw.upper()
    if upper in REASON_CODE_TO_CAPABILITY:
        return REASON_CODE_TO_CAPABILITY[upper]
    lower = raw.lower()
    try:
        return CapabilityId(lower)
    except ValueError:
        return CapabilityId.UNKNOWN


def is_round1_trainable(capability: str | CapabilityId | None) -> bool:
    return parse_capability_id(capability) in ROUND1_ENABLED_CAPABILITIES


def default_module_for(capability: str | CapabilityId | None) -> str:
    cap = parse_capability_id(capability)
    return CAPABILITY_DEFAULT_MODULE.get(cap, "unknown")
