"""Information-safe + executability gates for SCOPE Artifact V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.artifacts.provenance import (
    ALLOWED_RUNTIME_FIELDS,
    FORBIDDEN_ARTIFACT_KEYS,
    scan_dict_for_forbidden,
)
from harness.artifacts.schema import PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import CapabilityId, parse_capability_id
from harness.capability.state import DecisionState
from harness.telemetry.state_hash import env_purity_fingerprint, fingerprints_equal


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "details": dict(self.details)}


@dataclass(frozen=True)
class InformationSafeReport:
    visible: bool
    schema_valid: bool
    module_valid: bool
    executable: bool
    provenance_ok: bool
    purity_ok: bool
    gates: tuple[GateResult, ...]
    audit_error: str | None = None

    @property
    def all_passed(self) -> bool:
        return all(
            [
                self.visible,
                self.schema_valid,
                self.module_valid,
                self.executable,
                self.provenance_ok,
                self.purity_ok,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": self.visible,
            "schema_valid": self.schema_valid,
            "module_valid": self.module_valid,
            "executable": self.executable,
            "provenance_ok": self.provenance_ok,
            "purity_ok": self.purity_ok,
            "all_passed": self.all_passed,
            "audit_error": self.audit_error,
            "gates": [g.to_dict() for g in self.gates],
        }


# Capability → allowed recommended operations
CAPABILITY_ALLOWED_OPS: dict[CapabilityId, frozenset[str]] = {
    CapabilityId.DUPLICATE_EVIDENCE: frozenset(
        {
            "skip_curate",
            "replace_evidence",
            "KEEP_EVIDENCE",
            "SKIP_DUPLICATE",
            "curate_document",
            CapabilityActionType.CURATE_DOCUMENT.value,
            CapabilityActionType.UPDATE_EVIDENCE.value,
            CapabilityActionType.CONTINUE_SEARCH.value,
        }
    ),
    CapabilityId.PREMATURE_STOP: frozenset(
        {
            "continue_search",
            "verify",
            "open_source",
            "stop_and_answer",
            CapabilityActionType.CONTINUE_SEARCH.value,
            CapabilityActionType.SEARCH.value,
            CapabilityActionType.REWRITE_QUERY.value,
            CapabilityActionType.VERIFY_CLAIM.value,
            CapabilityActionType.OPEN_DOCUMENT.value,
            CapabilityActionType.STOP_AND_ANSWER.value,
        }
    ),
    CapabilityId.IRRELEVANT_EVIDENCE: frozenset(
        {
            "curate_document",
            CapabilityActionType.CURATE_DOCUMENT.value,
            CapabilityActionType.UPDATE_EVIDENCE.value,
        }
    ),
    CapabilityId.INVALID_CITATION: frozenset(
        {
            CapabilityActionType.OPEN_DOCUMENT.value,
            CapabilityActionType.CURATE_DOCUMENT.value,
            CapabilityActionType.VERIFY_CLAIM.value,
        }
    ),
}

FORBIDDEN_OPS_FOR_CAPABILITY: dict[CapabilityId, frozenset[str]] = {
    CapabilityId.DUPLICATE_EVIDENCE: frozenset(
        {
            CapabilityActionType.ANSWER.value,
            CapabilityActionType.STOP_AND_ANSWER.value,
            "final_answer",
            "invent_query_answer",
        }
    ),
    CapabilityId.PREMATURE_STOP: frozenset(
        {
            "invent_factual_answer",
            "emit_hidden_answer",
        }
    ),
}


def visibility_gate(
    state: DecisionState,
    artifact: PrivilegedArtifact,
) -> GateResult:
    unknown: list[str] = []
    observed = set(state.observed_ids)
    for eid in artifact.evidence_ids:
        if eid not in observed:
            unknown.append(str(eid))
    visible_docs = set(state.visible_document_ids) | set(state.pool_document_ids)
    unknown_docs: list[str] = []
    for did in artifact.document_ids:
        if did not in visible_docs:
            unknown_docs.append(str(did))
    passed = not unknown and not unknown_docs
    return GateResult(
        name="visibility",
        passed=passed,
        details={
            "visible": passed,
            "unknown_evidence_ids": unknown,
            "unknown_document_ids": unknown_docs,
        },
    )


def runtime_provenance_gate(artifact: PrivilegedArtifact) -> GateResult:
    bad = [f for f in artifact.runtime_fields_used if f not in ALLOWED_RUNTIME_FIELDS]
    # Also scan metadata / operation_args for forbidden keys
    forbidden_hits = scan_dict_for_forbidden(artifact.to_dict())
    passed = not bad and not forbidden_hits
    return GateResult(
        name="runtime_provenance",
        passed=passed,
        details={
            "invalid_runtime_fields": bad,
            "forbidden_keys": forbidden_hits,
        },
    )


def module_responsibility_gate(artifact: PrivilegedArtifact) -> GateResult:
    cap = artifact.resolved_capability()
    op = artifact.recommended_operation or (
        artifact.recommended_action.action_type.value
        if artifact.recommended_action
        else ""
    )
    if not op:
        # Endorse with no recommended op is ok
        if artifact.mode.value == "endorse":
            return GateResult(
                name="module_responsibility",
                passed=True,
                details={"operation": op, "capability_id": cap.value},
            )
        return GateResult(
            name="module_responsibility",
            passed=False,
            details={"error": "missing_operation", "capability_id": cap.value},
        )

    forbidden = FORBIDDEN_OPS_FOR_CAPABILITY.get(cap, frozenset())
    if op in forbidden:
        return GateResult(
            name="module_responsibility",
            passed=False,
            details={"error": "forbidden_operation", "operation": op, "capability_id": cap.value},
        )

    allowed = CAPABILITY_ALLOWED_OPS.get(cap)
    if allowed is not None and op not in allowed:
        return GateResult(
            name="module_responsibility",
            passed=False,
            details={"error": "operation_not_in_allowlist", "operation": op, "capability_id": cap.value},
        )
    return GateResult(
        name="module_responsibility",
        passed=True,
        details={"operation": op, "capability_id": cap.value},
    )


def executability_gate(
    state: DecisionState,
    artifact: PrivilegedArtifact,
    action: CapabilityAction | None = None,
) -> GateResult:
    """Validate recommended_operation + args as a CapabilityAction schema."""
    candidate = action or artifact.recommended_action
    if candidate is None and artifact.mode.value == "endorse":
        return GateResult(name="executability", passed=True, details={"skipped": "endorse_no_op"})
    if candidate is None:
        # Try to build from recommended_operation
        op = artifact.recommended_operation
        if not op:
            return GateResult(
                name="executability",
                passed=False,
                details={"error": "no_candidate_action"},
            )
        try:
            at = CapabilityActionType(op)
        except ValueError:
            # Map aliases
            alias = {
                "skip_curate": CapabilityActionType.CURATE_DOCUMENT,
                "replace_evidence": CapabilityActionType.CURATE_DOCUMENT,
                "continue_search": CapabilityActionType.CONTINUE_SEARCH,
                "verify": CapabilityActionType.VERIFY_CLAIM,
                "open_source": CapabilityActionType.OPEN_DOCUMENT,
                "stop_and_answer": CapabilityActionType.STOP_AND_ANSWER,
            }.get(op)
            if alias is None:
                return GateResult(
                    name="executability",
                    passed=False,
                    details={"error": f"unknown_operation:{op}"},
                )
            at = alias
        candidate = CapabilityAction(
            action_type=at,
            arguments=dict(artifact.operation_args),
            target_claim_id=artifact.target_claim_id,
        )

    reasons: list[str] = []
    if candidate.action_type == CapabilityActionType.UNKNOWN:
        reasons.append("unknown_action_type")
    if candidate.action_type == CapabilityActionType.SEARCH:
        q = candidate.arguments.get("query") or candidate.arguments.get("queries")
        if not q and artifact.operation_args.get("query_intent"):
            # Intent-only is allowed before realizer fills query
            pass
        elif not q:
            reasons.append("search_missing_query")
    if candidate.action_type == CapabilityActionType.VERIFY_CLAIM:
        if not str(candidate.arguments.get("claim", "")).strip() and not artifact.target_claim_id:
            reasons.append("verify_missing_claim")
    if candidate.action_type == CapabilityActionType.CURATE_DOCUMENT:
        # skip_curate may have empty add_ids
        pass

    # Action must not reference invisible docs
    visible_docs = set(state.visible_document_ids) | set(state.pool_document_ids)
    for key in ("doc_ids", "add_ids", "remove_ids", "document_ids"):
        vals = candidate.arguments.get(key)
        if isinstance(vals, list):
            for d in vals:
                if str(d) not in visible_docs and key != "remove_ids":
                    # remove_ids may reference curated that are visible
                    if str(d) not in set(state.curated_document_ids) | visible_docs:
                        reasons.append(f"doc_not_visible:{d}")

    return GateResult(
        name="executability",
        passed=not reasons,
        details={"reasons": reasons, "action_type": candidate.action_type.value},
    )


def schema_gate(artifact: PrivilegedArtifact) -> GateResult:
    reasons: list[str] = []
    if not artifact.schema_version:
        reasons.append("missing_schema_version")
    if not artifact.episode_id:
        reasons.append("missing_episode_id")
    if artifact.mode is None:
        reasons.append("missing_mode")
    if not artifact.reason_code:
        reasons.append("missing_reason_code")
    # Forbid teacher trace fields in metadata
    for k in FORBIDDEN_ARTIFACT_KEYS:
        if k in artifact.metadata:
            reasons.append(f"forbidden_metadata:{k}")
    return GateResult(name="schema", passed=not reasons, details={"reasons": reasons})


def shadow_purity_gate(
    fingerprint_before: dict[str, Any] | None,
    fingerprint_after: dict[str, Any] | None,
) -> GateResult:
    if fingerprint_before is None or fingerprint_after is None:
        return GateResult(
            name="shadow_purity",
            passed=True,
            details={"skipped": True, "note": "no fingerprints provided"},
        )
    ok = fingerprints_equal(fingerprint_before, fingerprint_after)
    return GateResult(
        name="shadow_purity",
        passed=ok,
        details={
            "before": fingerprint_before,
            "after": fingerprint_after,
            "audit_error": None if ok else "SHADOW_MUTATED_ENV",
        },
    )


def run_information_safe_gates(
    state: DecisionState,
    artifact: PrivilegedArtifact,
    *,
    candidate_action: CapabilityAction | None = None,
    fingerprint_before: dict[str, Any] | None = None,
    fingerprint_after: dict[str, Any] | None = None,
) -> InformationSafeReport:
    gates = (
        schema_gate(artifact),
        visibility_gate(state, artifact),
        runtime_provenance_gate(artifact),
        module_responsibility_gate(artifact),
        executability_gate(state, artifact, candidate_action),
        shadow_purity_gate(fingerprint_before, fingerprint_after),
    )
    by_name = {g.name: g for g in gates}
    purity = by_name["shadow_purity"]
    audit_error = None
    if not purity.passed:
        audit_error = "SHADOW_MUTATED_ENV"
    elif not by_name["visibility"].passed:
        audit_error = "VISIBILITY_VIOLATION"
    elif not by_name["runtime_provenance"].passed:
        audit_error = "PROVENANCE_VIOLATION"
    elif not by_name["module_responsibility"].passed:
        audit_error = "MODULE_RESPONSIBILITY_VIOLATION"
    elif not by_name["executability"].passed:
        audit_error = "EXECUTABILITY_VIOLATION"
    elif not by_name["schema"].passed:
        audit_error = "SCHEMA_INVALID"

    return InformationSafeReport(
        visible=by_name["visibility"].passed,
        schema_valid=by_name["schema"].passed,
        module_valid=by_name["module_responsibility"].passed,
        executable=by_name["executability"].passed,
        provenance_ok=by_name["runtime_provenance"].passed,
        purity_ok=purity.passed,
        gates=gates,
        audit_error=audit_error,
    )


def capture_env_fingerprint(env: Any) -> dict[str, Any]:
    return env_purity_fingerprint(env)
