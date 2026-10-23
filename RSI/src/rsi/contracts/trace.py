"""Traces, decision records and training-row provenance.

``DecisionRecord`` is the audit backbone of HarnessOPD (task book §8.5): the
exact token prefixes fed to the model on both sides, the alignment route, the
realizability/provenance reports and full identity digests.  Training must use
the recorded prefixes, never a re-assembled approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.action import Action
from rsi.contracts.events import Event

ALIGNMENT_ROUTES = ("explicit_map", "native_anchor", "skip", "runtime_only")


@dataclass(frozen=True)
class RealizabilityReport:
    """Three-layer realizability check (task book §8.3)."""

    syntax_legal: bool
    precondition_satisfied: bool
    information_realizable: bool
    reasons: tuple[str, ...] = ()
    # If information is not yet realizable, a suggested legal basic action to
    # acquire the missing observation first (route: acquire-then-supervise).
    acquire_first_action: Action | None = None

    @property
    def ok(self) -> bool:
        return self.syntax_legal and self.precondition_satisfied and self.information_realizable

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "syntax_legal": self.syntax_legal,
            "precondition_satisfied": self.precondition_satisfied,
            "information_realizable": self.information_realizable,
            "reasons": list(self.reasons),
            "acquire_first_action": self.acquire_first_action.to_json() if self.acquire_first_action else None,
        }


@dataclass(frozen=True)
class ProvenanceReport:
    """Information-flow check: which external facts the target action needs and
    where each comes from.  Engineering checks, not a formal guarantee."""

    required_fact_refs: tuple[str, ...] = ()
    satisfied_by_student_history: tuple[str, ...] = ()
    leaked_refs: tuple[str, ...] = ()
    verdict: str = "clean"  # clean | leak | uncertain

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "required_fact_refs": list(self.required_fact_refs),
            "satisfied_by_student_history": list(self.satisfied_by_student_history),
            "leaked_refs": list(self.leaked_refs),
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of aligning one Teacher branch to Student-native supervision."""

    route: str  # one of ALIGNMENT_ROUTES
    target_action: Action | None
    source_event_ids: tuple[str, ...]
    anchor_index: int | None
    student_pre_state_digest: str
    teacher_context_ref: str | None = None
    legality: RealizabilityReport | None = None
    provenance: ProvenanceReport | None = None
    reject_reason: str | None = None
    # Recipes project a* that is not the Teacher's original sampled action; its
    # score must be labeled teacher_scored_projected_target (§8.8).
    target_kind: str = "native"  # native | teacher_scored_projected_target

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "route": self.route,
            "target_action": self.target_action.to_json() if self.target_action else None,
            "source_event_ids": list(self.source_event_ids),
            "anchor_index": self.anchor_index,
            "student_pre_state_digest": self.student_pre_state_digest,
            "teacher_context_ref": self.teacher_context_ref,
            "legality": self.legality.to_json() if self.legality else None,
            "provenance": self.provenance.to_json() if self.provenance else None,
            "reject_reason": self.reject_reason,
            "target_kind": self.target_kind,
        }


@dataclass(frozen=True)
class DecisionRecord:
    """One Student decision point, with optional aligned OPD target."""

    campaign_id: str
    round_index: int
    batch_index: int
    episode_id: str
    decision_index: int
    environment_snapshot_ref: str
    public_view_digest: str
    runtime_contract_digest: str
    policy_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    bank_digest: str
    # Exact token ids that entered the model (Student side).
    student_prompt_token_ids: tuple[int, ...]
    sampled_action_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    action_mask: tuple[int, ...]
    # OPD target side (present iff an alignment produced supervision).
    teacher_prompt_token_ids: tuple[int, ...] = ()
    target_action: Action | None = None
    target_token_ids: tuple[int, ...] = ()
    target_mask: tuple[int, ...] = ()
    alignment: AlignmentResult | None = None
    source_event_ids: tuple[str, ...] = ()
    reward_ref: str | None = None
    behavior_policy_version: str = ""
    scorer_version: str = ""
    sampling_parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    extra: Mapping[str, JSONValue] = field(default_factory=dict)

    def record_id(self) -> str:
        return digest_of(
            {
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "batch_index": self.batch_index,
                "episode_id": self.episode_id,
                "decision_index": self.decision_index,
            }
        )

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "record_id": self.record_id(),
            "campaign_id": self.campaign_id,
            "round_index": self.round_index,
            "batch_index": self.batch_index,
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "environment_snapshot_ref": self.environment_snapshot_ref,
            "public_view_digest": self.public_view_digest,
            "runtime_contract_digest": self.runtime_contract_digest,
            "policy_digest": self.policy_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "chat_template_digest": self.chat_template_digest,
            "bank_digest": self.bank_digest,
            "student_prompt_token_ids": list(self.student_prompt_token_ids),
            "sampled_action_token_ids": list(self.sampled_action_token_ids),
            "old_logprobs": list(self.old_logprobs),
            "action_mask": list(self.action_mask),
            "teacher_prompt_token_ids": list(self.teacher_prompt_token_ids),
            "target_action": self.target_action.to_json() if self.target_action else None,
            "target_token_ids": list(self.target_token_ids),
            "target_mask": list(self.target_mask),
            "alignment": self.alignment.to_json() if self.alignment else None,
            "source_event_ids": list(self.source_event_ids),
            "reward_ref": self.reward_ref,
            "behavior_policy_version": self.behavior_policy_version,
            "scorer_version": self.scorer_version,
            "sampling_parameters": dict(self.sampling_parameters),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class EpisodeTrace:
    """Immutable raw record of one episode: full event log + decisions."""

    episode_id: str
    task_id: str
    domain: str
    policy_digest: str
    bank_digest: str
    events: tuple[Event, ...]
    decisions: tuple[DecisionRecord, ...]
    final_answer: str | None
    outcome_reward: float | None
    terminated_reason: str
    synthetic: bool = False

    def total_decision_count(self) -> int:
        return len(self.decisions)
