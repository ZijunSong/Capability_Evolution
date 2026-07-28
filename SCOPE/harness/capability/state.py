"""Immutable DecisionState V2 and observation lineage types for SCOPE.

V2 adds provenance-friendly fields while keeping V1 field names for
backward compatibility with existing audit / shadow / drivers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from harness.telemetry.state_hash import hash_decision_state_core

SCHEMA_VERSION = "scope.decision_state.v2"
SCHEMA_VERSION_V1 = "decision_state.v1"

SourceType = Literal["search", "grep", "read", "review", "verify", "curate", "other"]


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    source_type: SourceType
    source_document_ids: tuple[str, ...]
    created_turn: int
    visible_to_student: bool
    text_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationRecord":
        return cls(
            observation_id=str(data["observation_id"]),
            source_type=data["source_type"],
            source_document_ids=tuple(data.get("source_document_ids", ())),
            created_turn=int(data["created_turn"]),
            visible_to_student=bool(data["visible_to_student"]),
            text_hash=str(data["text_hash"]),
        )


@dataclass(frozen=True)
class ActionRecord:
    turn_id: int
    action_type: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "action_type": self.action_type,
            "arguments": dict(self.arguments),
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionRecord":
        return cls(
            turn_id=int(data["turn_id"]),
            action_type=str(data["action_type"]),
            arguments=dict(data.get("arguments", {})),
            raw_text=str(data.get("raw_text", "")),
        )


@dataclass(frozen=True)
class ClaimState:
    claim_id: str
    text: str
    status: str
    supporting_document_ids: tuple[str, ...] = ()
    source_observation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimState":
        return cls(
            claim_id=str(data["claim_id"]),
            text=str(data["text"]),
            status=str(data.get("status", "unknown")),
            supporting_document_ids=tuple(data.get("supporting_document_ids", ())),
            source_observation_ids=tuple(data.get("source_observation_ids", ())),
        )


@dataclass(frozen=True)
class VerificationRecordState:
    turn_id: int
    claim: str
    document_ids: tuple[str, ...]
    judgments: dict[str, bool]
    source_observation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "claim": self.claim,
            "document_ids": list(self.document_ids),
            "judgments": dict(self.judgments),
            "source_observation_ids": list(self.source_observation_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationRecordState":
        return cls(
            turn_id=int(data["turn_id"]),
            claim=str(data["claim"]),
            document_ids=tuple(data.get("document_ids", ())),
            judgments={str(k): bool(v) for k, v in dict(data.get("judgments", {})).items()},
            source_observation_ids=tuple(data.get("source_observation_ids", ())),
        )


def _derive_claim_buckets(
    evidence_claims: tuple[ClaimState, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    supported: list[str] = []
    unsupported: list[str] = []
    conflicting: list[str] = []
    for c in evidence_claims:
        st = str(c.status).lower()
        if st in {"supported", "verified", "linked"}:
            supported.append(c.claim_id)
        elif st in {"conflict", "conflicting", "contradicted"}:
            conflicting.append(c.claim_id)
        elif st in {"unsupported", "weak", "unverified", "partial", "unknown", ""}:
            unsupported.append(c.claim_id)
        else:
            unsupported.append(c.claim_id)
    return tuple(supported), tuple(unsupported), tuple(conflicting)


@dataclass(frozen=True)
class DecisionState:
    """DecisionState V2 (alias DecisionStateV2).

    V1 fields retained. New V2 fields have defaults so existing constructors
    continue to work.
    """

    episode_id: str
    task_id: str
    turn_id: int

    query: str
    rendered_context: str

    action_history: tuple[ActionRecord, ...]
    observation_ids: tuple[str, ...]
    visible_document_ids: tuple[str, ...]

    pool_document_ids: tuple[str, ...]
    curated_document_ids: tuple[str, ...]
    evidence_claims: tuple[ClaimState, ...]
    verification_records: tuple[VerificationRecordState, ...]

    remaining_turns: int
    remaining_search_calls: int | None
    token_budget_used: int
    token_budget_total: int

    last_action_type: str | None
    repeated_query_score: float | None

    wm_snapshot_hash: str
    schema_version: str = SCHEMA_VERSION

    # --- V2 additions ---
    event_id: str = ""
    goal: str = ""
    active_subgoal: str = ""
    last_action_arguments: dict[str, Any] = field(default_factory=dict)
    last_query: str = ""
    repeated_query_count: int = 0
    remaining_open_calls: int | None = None
    student_action: dict[str, Any] | None = None
    supported_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    conflicting_claims: tuple[str, ...] = ()

    @property
    def observed_ids(self) -> tuple[str, ...]:
        """Alias for observation_ids (V2 naming)."""
        return self.observation_ids

    @property
    def candidate_evidence_ids(self) -> tuple[str, ...]:
        return self.pool_document_ids

    @property
    def curated_evidence_ids(self) -> tuple[str, ...]:
        return self.curated_document_ids

    def with_derived_claims(self) -> "DecisionState":
        if self.supported_claims or self.unsupported_claims or self.conflicting_claims:
            return self
        sup, unsup, conf = _derive_claim_buckets(self.evidence_claims)
        from dataclasses import replace

        return replace(
            self,
            supported_claims=sup,
            unsupported_claims=unsup,
            conflicting_claims=conf,
            goal=self.goal or self.query,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "event_id": self.event_id,
            "query": self.query,
            "goal": self.goal or self.query,
            "active_subgoal": self.active_subgoal,
            "rendered_context": self.rendered_context,
            "action_history": [a.to_dict() for a in self.action_history],
            "observation_ids": list(self.observation_ids),
            "observed_ids": list(self.observation_ids),
            "visible_document_ids": list(self.visible_document_ids),
            "pool_document_ids": list(self.pool_document_ids),
            "candidate_evidence_ids": list(self.pool_document_ids),
            "curated_document_ids": list(self.curated_document_ids),
            "curated_evidence_ids": list(self.curated_document_ids),
            "evidence_claims": [c.to_dict() for c in self.evidence_claims],
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "conflicting_claims": list(self.conflicting_claims),
            "verification_records": [v.to_dict() for v in self.verification_records],
            "remaining_turns": self.remaining_turns,
            "remaining_search_calls": self.remaining_search_calls,
            "remaining_open_calls": self.remaining_open_calls,
            "token_budget_used": self.token_budget_used,
            "token_budget_total": self.token_budget_total,
            "last_action_type": self.last_action_type,
            "last_action_arguments": dict(self.last_action_arguments),
            "last_query": self.last_query,
            "repeated_query_score": self.repeated_query_score,
            "repeated_query_count": self.repeated_query_count,
            "student_action": dict(self.student_action) if self.student_action else None,
            "wm_snapshot_hash": self.wm_snapshot_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionState":
        obs = data.get("observation_ids") or data.get("observed_ids") or ()
        pool = data.get("pool_document_ids") or data.get("candidate_evidence_ids") or ()
        curated = data.get("curated_document_ids") or data.get("curated_evidence_ids") or ()
        student = data.get("student_action")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            episode_id=str(data["episode_id"]),
            task_id=str(data["task_id"]),
            turn_id=int(data["turn_id"]),
            event_id=str(data.get("event_id", "")),
            query=str(data["query"]),
            goal=str(data.get("goal") or data.get("query", "")),
            active_subgoal=str(data.get("active_subgoal", "")),
            rendered_context=str(data.get("rendered_context", "")),
            action_history=tuple(
                ActionRecord.from_dict(x) for x in data.get("action_history", [])
            ),
            observation_ids=tuple(obs),
            visible_document_ids=tuple(data.get("visible_document_ids", ())),
            pool_document_ids=tuple(pool),
            curated_document_ids=tuple(curated),
            evidence_claims=tuple(
                ClaimState.from_dict(x) for x in data.get("evidence_claims", [])
            ),
            verification_records=tuple(
                VerificationRecordState.from_dict(x)
                for x in data.get("verification_records", [])
            ),
            remaining_turns=int(data.get("remaining_turns", 0)),
            remaining_search_calls=data.get("remaining_search_calls"),
            remaining_open_calls=data.get("remaining_open_calls"),
            token_budget_used=int(data.get("token_budget_used", 0)),
            token_budget_total=int(data.get("token_budget_total", 0)),
            last_action_type=data.get("last_action_type"),
            last_action_arguments=dict(data.get("last_action_arguments") or {}),
            last_query=str(data.get("last_query", "")),
            repeated_query_score=data.get("repeated_query_score"),
            repeated_query_count=int(data.get("repeated_query_count", 0) or 0),
            student_action=dict(student) if isinstance(student, dict) else None,
            supported_claims=tuple(data.get("supported_claims", ())),
            unsupported_claims=tuple(data.get("unsupported_claims", ())),
            conflicting_claims=tuple(data.get("conflicting_claims", ())),
            wm_snapshot_hash=str(data.get("wm_snapshot_hash", "")),
        )

    def state_hash(self) -> str:
        """Full JSON hash (legacy). Prefer core_state_hash for purity audits."""
        payload = self.to_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def core_state_hash(self) -> str:
        return hash_decision_state_core(self.to_dict())

    def check_info_safety(self) -> tuple[bool, tuple[str, ...]]:
        """Info(DecisionState) ⊆ Info(student runtime): no forbidden fields."""
        from harness.artifacts.provenance import assert_info_subset

        return assert_info_subset(self.to_dict().keys())

    def field_provenance(self) -> dict[str, str]:
        from harness.artifacts.provenance import (
            DECISION_STATE_FIELD_PROVENANCE,
            ProvenanceKind,
        )

        return {
            k: DECISION_STATE_FIELD_PROVENANCE.get(
                k, ProvenanceKind.PRIVILEGED_FORBIDDEN
            ).value
            for k in self.to_dict().keys()
        }


# Explicit V2 alias
DecisionStateV2 = DecisionState


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute_wm_snapshot_hash(
    *,
    curated_ids: list[str] | tuple[str, ...] | None = None,
    pool_ids: list[str] | tuple[str, ...] | None = None,
    search_history: list[str] | tuple[str, ...] | None = None,
    turn_number: int = 0,
    observation_ids: list[str] | tuple[str, ...] = (),
) -> str:
    payload = json.dumps(
        {
            "turn": turn_number,
            "curated": list(curated_ids or ()),
            "pool": list(pool_ids or ()),
            "history": list(search_history or ()),
            "obs": list(observation_ids),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
