"""Rule-based conditional module router (v1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterInput:
    query_length: int
    constraint_count: int
    multi_hop_likelihood: float
    evidence_count: int
    conflicting_evidence_count: int
    remaining_budget: int
    model_confidence: float
    repeated_search_count: int


def should_activate_verification(features: RouterInput) -> bool:
    multi_constraint = features.constraint_count >= 2
    conflicting = features.conflicting_evidence_count > 0
    low_confidence = features.model_confidence < 0.5
    return multi_constraint or conflicting or low_confidence


def should_activate_evidence_state(features: RouterInput) -> bool:
    return features.evidence_count > 10 or features.multi_hop_likelihood > 0.5


def should_activate_budget_control(features: RouterInput) -> bool:
    return features.remaining_budget < 8 or features.repeated_search_count >= 3
