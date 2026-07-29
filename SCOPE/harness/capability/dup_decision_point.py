"""Duplicate capability decision point metadata (Round 3).

Represents *Capability Decision Point Selection* — where duplicate_evidence
should intervene — distinct from shadow KEEP/SKIP judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState


CAPABILITY_ID_DUPLICATE = "duplicate_evidence"
DECISION_TYPE_EVIDENCE_ADMISSION = "evidence_admission"


@dataclass(frozen=True)
class DupDecisionPoint:
    """Typed decision point at evidence admission."""

    capability_id: str = CAPABILITY_ID_DUPLICATE
    decision_type: str = DECISION_TYPE_EVIDENCE_ADMISSION
    candidate_evidence_id: str = ""
    candidate_source_id: str | None = None
    observed_ids: tuple[str, ...] = ()
    curated_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "decision_type": self.decision_type,
            "candidate_evidence_id": self.candidate_evidence_id,
            "candidate_source_id": self.candidate_source_id,
            "observed_ids": list(self.observed_ids),
            "curated_evidence_ids": list(self.curated_evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DupDecisionPoint":
        return cls(
            capability_id=str(data.get("capability_id", CAPABILITY_ID_DUPLICATE)),
            decision_type=str(
                data.get("decision_type", DECISION_TYPE_EVIDENCE_ADMISSION)
            ),
            candidate_evidence_id=str(data.get("candidate_evidence_id", "")),
            candidate_source_id=data.get("candidate_source_id"),
            observed_ids=tuple(data.get("observed_ids") or ()),
            curated_evidence_ids=tuple(data.get("curated_evidence_ids") or ()),
        )


def is_evidence_admission_action(action: CapabilityAction) -> bool:
    """Student is about to admit evidence via curate."""
    return action.action_type == CapabilityActionType.CURATE_DOCUMENT


def extract_curate_candidates(
    state: DecisionState,
    student_action: CapabilityAction,
) -> list[str]:
    """Return pool-visible candidate ids from a curate action."""
    if not is_evidence_admission_action(student_action):
        return []
    pool = set(state.pool_document_ids) | set(state.visible_document_ids)
    add_ids = student_action.arguments.get("add_ids") or []
    if not isinstance(add_ids, list):
        return []
    return [str(d) for d in add_ids if str(d) in pool]


def build_decision_points(
    state: DecisionState,
    student_action: CapabilityAction,
) -> list[DupDecisionPoint]:
    """One decision point per curate candidate the student actually proposed."""
    curated = tuple(state.curated_document_ids)
    observed = tuple(state.observation_ids)
    points: list[DupDecisionPoint] = []
    for cid in extract_curate_candidates(state, student_action):
        points.append(
            DupDecisionPoint(
                candidate_evidence_id=cid,
                candidate_source_id=cid,
                observed_ids=observed,
                curated_evidence_ids=curated,
            )
        )
    return points


def is_duplicate_candidate(
    candidate_id: str,
    curated_ids: tuple[str, ...] | list[str],
) -> bool:
    """Candidate is duplicate/near-duplicate if already in curated set."""
    return str(candidate_id) in {str(c) for c in curated_ids}
