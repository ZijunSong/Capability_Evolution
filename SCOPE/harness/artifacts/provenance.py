"""Field provenance taxonomy for DecisionState / Artifact information safety."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class ProvenanceKind(str, Enum):
    OBSERVED = "OBSERVED"
    RUNTIME = "RUNTIME"
    DERIVED_VISIBLE = "DERIVED_VISIBLE"
    PRIVILEGED_FORBIDDEN = "PRIVILEGED_FORBIDDEN"


# Canonical DecisionState field → provenance
DECISION_STATE_FIELD_PROVENANCE: dict[str, ProvenanceKind] = {
    # identity
    "episode_id": ProvenanceKind.RUNTIME,
    "task_id": ProvenanceKind.RUNTIME,
    "turn_id": ProvenanceKind.RUNTIME,
    "event_id": ProvenanceKind.RUNTIME,
    "schema_version": ProvenanceKind.RUNTIME,
    # goals
    "query": ProvenanceKind.OBSERVED,
    "goal": ProvenanceKind.OBSERVED,
    "active_subgoal": ProvenanceKind.DERIVED_VISIBLE,
    "rendered_context": ProvenanceKind.OBSERVED,
    # history / actions
    "action_history": ProvenanceKind.OBSERVED,
    "last_action_type": ProvenanceKind.OBSERVED,
    "last_action_arguments": ProvenanceKind.OBSERVED,
    "last_query": ProvenanceKind.OBSERVED,
    "student_action": ProvenanceKind.OBSERVED,
    # evidence / observations
    "observation_ids": ProvenanceKind.OBSERVED,
    "observed_ids": ProvenanceKind.OBSERVED,
    "visible_document_ids": ProvenanceKind.OBSERVED,
    "pool_document_ids": ProvenanceKind.OBSERVED,
    "candidate_evidence_ids": ProvenanceKind.OBSERVED,
    "curated_document_ids": ProvenanceKind.OBSERVED,
    "curated_evidence_ids": ProvenanceKind.OBSERVED,
    "evidence_claims": ProvenanceKind.OBSERVED,
    "supported_claims": ProvenanceKind.DERIVED_VISIBLE,
    "unsupported_claims": ProvenanceKind.DERIVED_VISIBLE,
    "conflicting_claims": ProvenanceKind.DERIVED_VISIBLE,
    "verification_records": ProvenanceKind.OBSERVED,
    # runtime counters
    "remaining_turns": ProvenanceKind.RUNTIME,
    "remaining_search_calls": ProvenanceKind.RUNTIME,
    "remaining_open_calls": ProvenanceKind.RUNTIME,
    "token_budget_used": ProvenanceKind.RUNTIME,
    "token_budget_total": ProvenanceKind.RUNTIME,
    "repeated_query_score": ProvenanceKind.DERIVED_VISIBLE,
    "repeated_query_count": ProvenanceKind.DERIVED_VISIBLE,
    "wm_snapshot_hash": ProvenanceKind.DERIVED_VISIBLE,
    # forbidden if ever present
    "gold_answer": ProvenanceKind.PRIVILEGED_FORBIDDEN,
    "hidden_relevance_label": ProvenanceKind.PRIVILEGED_FORBIDDEN,
    "teacher_trace": ProvenanceKind.PRIVILEGED_FORBIDDEN,
    "teacher_completion": ProvenanceKind.PRIVILEGED_FORBIDDEN,
    "hidden_verifier_text": ProvenanceKind.PRIVILEGED_FORBIDDEN,
    "future_observation_ids": ProvenanceKind.PRIVILEGED_FORBIDDEN,
}

# Runtime fields that shadow may *reference* (must come from DecisionState, not invent)
ALLOWED_RUNTIME_FIELDS: frozenset[str] = frozenset(
    {
        "remaining_turns",
        "remaining_search_calls",
        "remaining_open_calls",
        "token_budget_used",
        "token_budget_total",
        "repeated_query_score",
        "repeated_query_count",
        "turn_id",
        "episode_id",
        "task_id",
    }
)

FORBIDDEN_ARTIFACT_KEYS: frozenset[str] = frozenset(
    {
        "gold_answer",
        "teacher_trace",
        "teacher_completion",
        "teacher_trajectory",
        "hidden_verifier_text",
        "hidden_relevance_label",
        "future_observation_ids",
        "future_observations",
        "hidden_answer",
    }
)


def provenance_of(field_name: str) -> ProvenanceKind:
    return DECISION_STATE_FIELD_PROVENANCE.get(
        field_name, ProvenanceKind.PRIVILEGED_FORBIDDEN
    )


def is_student_visible_field(field_name: str) -> bool:
    kind = provenance_of(field_name)
    return kind in {
        ProvenanceKind.OBSERVED,
        ProvenanceKind.RUNTIME,
        ProvenanceKind.DERIVED_VISIBLE,
    }


def assert_info_subset(
    field_names: Iterable[str],
) -> tuple[bool, tuple[str, ...]]:
    """Check Info(fields) ⊆ Info(student runtime state)."""
    bad = [
        f
        for f in field_names
        if provenance_of(f) == ProvenanceKind.PRIVILEGED_FORBIDDEN
    ]
    return (len(bad) == 0, tuple(bad))


def scan_dict_for_forbidden(payload: dict[str, Any], *, prefix: str = "") -> list[str]:
    """Recursively flag forbidden privileged keys in nested dicts."""
    hits: list[str] = []
    for k, v in payload.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if str(k) in FORBIDDEN_ARTIFACT_KEYS:
            hits.append(path)
        if isinstance(v, dict):
            hits.extend(scan_dict_for_forbidden(v, prefix=path))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    hits.extend(scan_dict_for_forbidden(item, prefix=f"{path}[{i}]"))
    return hits
