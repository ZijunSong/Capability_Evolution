"""A4: Verification gate ablation at data/prediction layer only.

ActionRealizer hard constraints are NEVER removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GateFlags:
    visibility: bool = True
    schema_exec: bool = True
    mutation: bool = True
    verification: bool = True
    accept_endorse_correct: bool = True


VARIANT_FLAGS = {
    "a4_full_gate": GateFlags(),
    "a4_no_visibility_gate": GateFlags(visibility=False),
    "a4_no_schema_exec_gate": GateFlags(schema_exec=False),
    "a4_no_mutation_gate": GateFlags(mutation=False),
    "a4_no_verification": GateFlags(
        visibility=False, schema_exec=False, mutation=False, verification=False
    ),
    "a4_accept_only_no_endorse_correct": GateFlags(accept_endorse_correct=False),
}


@dataclass
class GateTelemetry:
    accepted: int = 0
    rejected: int = 0
    visibility_violation: int = 0
    schema_invalid: int = 0
    unexecutable_operation: int = 0
    state_mutation: int = 0
    label_disagreement: int = 0
    invalid_live_action: int = 0
    false_intervention: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "visibility_violation": self.visibility_violation,
            "schema_invalid": self.schema_invalid,
            "unexecutable_operation": self.unexecutable_operation,
            "state_mutation": self.state_mutation,
            "label_disagreement": self.label_disagreement,
            "invalid_live_action": self.invalid_live_action,
            "false_intervention": self.false_intervention,
        }


def flags_for_variant(variant: str) -> GateFlags:
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"unknown A4 variant: {variant}")
    return VARIANT_FLAGS[variant]


def apply_gates(
    candidates: list[dict[str, Any]],
    flags: GateFlags,
    *,
    hard_realizer_check: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], GateTelemetry]:
    """Filter supervision/prediction candidates. Hard realizer always applied."""
    tel = GateTelemetry()
    kept: list[dict[str, Any]] = []
    for c in candidates:
        reasons: list[str] = []
        if flags.visibility and c.get("visibility_ok") is False:
            tel.visibility_violation += 1
            reasons.append("visibility")
        if flags.schema_exec and c.get("schema_ok") is False:
            tel.schema_invalid += 1
            reasons.append("schema")
        if flags.schema_exec and c.get("executable") is False:
            tel.unexecutable_operation += 1
            reasons.append("unexecutable")
        if flags.mutation and c.get("mutation_ok") is False:
            tel.state_mutation += 1
            reasons.append("mutation")
        if flags.verification and c.get("verified") is False:
            reasons.append("verification")
        if flags.accept_endorse_correct:
            route = c.get("route")
            if route not in (None, "ENDORSE", "CORRECT"):
                reasons.append("route")
        # Hard constraint — cannot be disabled by ablation.
        if not hard_realizer_check(c):
            tel.invalid_live_action += 1
            reasons.append("hard_realizer")
        if c.get("false_intervention"):
            tel.false_intervention += 1
        if c.get("label_disagreement"):
            tel.label_disagreement += 1

        if reasons:
            tel.rejected += 1
            tel.details.append({"id": c.get("id"), "rejected_by": reasons})
        else:
            tel.accepted += 1
            kept.append(c)
    return kept, tel
