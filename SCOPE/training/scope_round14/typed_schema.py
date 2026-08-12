"""Typed action / decision-state schemas for Round14 capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.capability.capability_id import CapabilityId

ROUND14_CAPABILITIES: tuple[str, ...] = (
    "duplicate_evidence",
    "stop_decision",
    "verification_routing",
    "evidence_admission",
    "context_budget_routing",
    "external_verification_routing",
    "rollback_lite",
)


@dataclass(frozen=True)
class CapabilitySchema:
    capability_id: str
    canonical_capability: CapabilityId
    actions: tuple[str, ...]
    decision_state_fields: tuple[str, ...]
    module_id: str
    harness_config_key: str
    description: str = ""


CAPABILITY_SCHEMAS: dict[str, CapabilitySchema] = {
    "duplicate_evidence": CapabilitySchema(
        capability_id="duplicate_evidence",
        canonical_capability=CapabilityId.DUPLICATE_EVIDENCE,
        actions=("KEEP_EVIDENCE", "SKIP_DUPLICATE"),
        decision_state_fields=(
            "curated_document_ids",
            "pool_document_ids",
            "visible_document_ids",
            "candidate_evidence_id",
        ),
        module_id="evidence_state",
        harness_config_key="content_dedup",
        description="C0 positive anchor — bilateral dup at curate decision points",
    ),
    "stop_decision": CapabilitySchema(
        capability_id="stop_decision",
        canonical_capability=CapabilityId.STOP_DECISION,
        actions=("STOP", "CONTINUE"),
        decision_state_fields=(
            "turn_index",
            "budget_remaining",
            "verification_records",
            "curated_document_ids",
        ),
        module_id="verification",
        harness_config_key="stop_budget_hint",
        description="C1 premature stop routing",
    ),
    "verification_routing": CapabilitySchema(
        capability_id="verification_routing",
        canonical_capability=CapabilityId.VERIFICATION_DECISION,
        actions=("VERIFY", "NO_VERIFY"),
        decision_state_fields=(
            "turn_index",
            "curated_document_ids",
            "pending_claims",
            "verification_records",
        ),
        module_id="verification",
        harness_config_key="verification_aware_curation",
        description="C2 when to verify (not execution)",
    ),
    "evidence_admission": CapabilitySchema(
        capability_id="evidence_admission",
        canonical_capability=CapabilityId.EVIDENCE_ADMISSION,
        actions=("ADMIT", "DROP"),
        decision_state_fields=(
            "candidate_evidence_id",
            "candidate_text",
            "curated_document_ids",
            "pool_document_ids",
        ),
        module_id="evidence_state",
        harness_config_key="subtractive_curation",
        description="C3 single-candidate admit/drop",
    ),
    "context_budget_routing": CapabilitySchema(
        capability_id="context_budget_routing",
        canonical_capability=CapabilityId.CONTEXT_BUDGET_ROUTING,
        actions=("KEEP_CONTEXT", "COMPRESS_OR_DROP"),
        decision_state_fields=(
            "token_estimate",
            "budget_remaining",
            "context_segments",
            "turn_index",
        ),
        module_id="context_budget",
        harness_config_key="sentence_compression",
        description="C4 routing only — hard truncation stays in runtime",
    ),
    "external_verification_routing": CapabilitySchema(
        capability_id="external_verification_routing",
        canonical_capability=CapabilityId.EXTERNAL_VERIFICATION,
        actions=("VERIFY_EXTERNALLY", "DO_NOT"),
        decision_state_fields=(
            "turn_index",
            "curated_document_ids",
            "pending_claims",
            "tool_schema_visible",
        ),
        module_id="verification",
        harness_config_key="expose_verify_tool",
        description="C5 route to external verify — execution stays runtime",
    ),
    "rollback_lite": CapabilitySchema(
        capability_id="rollback_lite",
        canonical_capability=CapabilityId.ROLLBACK_LITE,
        actions=("RECOVER", "CONTINUE"),
        decision_state_fields=(
            "turn_index",
            "failure_history",
            "recovery_budget",
            "previous_operation",
        ),
        module_id="recovery",
        harness_config_key="enabled",
        description="C6 binary recovery — no checkpoint ID selection",
    ),
}


def get_schema(capability: str) -> CapabilitySchema:
    key = capability.strip().lower()
    if key not in CAPABILITY_SCHEMAS:
        raise KeyError(f"unknown Round14 capability: {capability}")
    return CAPABILITY_SCHEMAS[key]


def validate_row(capability: str, row: dict[str, Any]) -> list[str]:
    """Return list of schema violations (empty = ok)."""
    schema = get_schema(capability)
    errors: list[str] = []
    label = (
        row.get("gold_action")
        or row.get("gold_operation")
        or row.get("operation")
        or row.get("target_action", {}).get("operation")
    )
    if label and str(label) not in schema.actions:
        errors.append(f"unknown action {label!r} for {capability}")
    ds = row.get("decision_state")
    if ds is not None and not isinstance(ds, dict):
        errors.append("decision_state must be dict")
    return errors
