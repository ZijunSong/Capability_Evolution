"""Training-row construction (task book §8.1).

Two row kinds that must never be confused:

- **RL row** — the Student's *real sampled action* under the exact H0 token
  prefix it saw; feeds CISPO/outcome RL.  Teacher actions never replace the
  sampled action; old logprobs are never mixed.
- **OPD row** — the aligned Teacher-derived Student action under (Student H0
  prefix + private Teacher prefix, stored separately); feeds the gated
  projected-gap loss.

``DecisionRecord`` fields are the audit backbone; rows carry references, not
re-assembled prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.action import Action
from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.trace import AlignmentResult, DecisionRecord
from rsi.opd.scoring import TargetScore


@dataclass(frozen=True)
class RLRow:
    row_id: str
    episode_id: str
    decision_index: int
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    action_mask: tuple[int, ...]
    advantage: float
    reward_ref: str
    behavior_policy_version: str

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "row_id": self.row_id,
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "prompt_token_ids": list(self.prompt_token_ids),
            "action_token_ids": list(self.action_token_ids),
            "old_logprobs": list(self.old_logprobs),
            "action_mask": list(self.action_mask),
            "advantage": self.advantage,
            "reward_ref": self.reward_ref,
            "behavior_policy_version": self.behavior_policy_version,
        }


@dataclass(frozen=True)
class OPDRow:
    row_id: str
    episode_id: str
    decision_index: int
    student_prompt_token_ids: tuple[int, ...]
    teacher_prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    target_mask: tuple[int, ...]
    target_action: Action | None
    target_kind: str
    alignment_route: str
    source_event_ids: tuple[str, ...]
    realizability_json: Mapping[str, JSONValue]
    provenance_json: Mapping[str, JSONValue]
    token_weight: tuple[float, ...] = ()
    # Teacher-forced logprobs of the target tokens on both sides (auditable).
    student_logprobs: tuple[float, ...] = ()
    teacher_logprobs: tuple[float, ...] = ()

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "row_id": self.row_id,
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "student_prompt_token_ids": list(self.student_prompt_token_ids),
            "teacher_prompt_token_ids": list(self.teacher_prompt_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "target_mask": list(self.target_mask),
            "target_action": self.target_action.to_json() if self.target_action else None,
            "target_kind": self.target_kind,
            "alignment_route": self.alignment_route,
            "source_event_ids": list(self.source_event_ids),
            "realizability": dict(self.realizability_json),
            "provenance": dict(self.provenance_json),
            "token_weight": list(self.token_weight),
            "student_logprobs": list(self.student_logprobs),
            "teacher_logprobs": list(self.teacher_logprobs),
        }


def build_rl_row(decision: DecisionRecord, *, advantage: float, reward_ref: str) -> RLRow:
    return RLRow(
        row_id=digest_of({"d": decision.record_id(), "kind": "rl"}),
        episode_id=decision.episode_id,
        decision_index=decision.decision_index,
        prompt_token_ids=tuple(decision.student_prompt_token_ids),
        action_token_ids=tuple(decision.sampled_action_token_ids),
        old_logprobs=tuple(decision.old_logprobs),
        action_mask=tuple(decision.action_mask),
        advantage=float(advantage),
        reward_ref=reward_ref,
        behavior_policy_version=decision.behavior_policy_version,
    )


def build_opd_row(decision: DecisionRecord, score: TargetScore, alignment: AlignmentResult) -> OPDRow:
    return OPDRow(
        row_id=digest_of({"d": decision.record_id(), "kind": "opd"}),
        episode_id=decision.episode_id,
        decision_index=decision.decision_index,
        student_prompt_token_ids=tuple(score.student_prompt_token_ids),
        teacher_prompt_token_ids=tuple(score.teacher_prompt_token_ids),
        target_token_ids=tuple(score.target_token_ids),
        target_mask=tuple(1 for _ in score.target_token_ids),
        target_action=alignment.target_action,
        target_kind=score.target_kind,
        alignment_route=alignment.route,
        source_event_ids=tuple(alignment.source_event_ids),
        realizability_json=alignment.legality.to_json() if alignment.legality else {},
        provenance_json=alignment.provenance.to_json() if alignment.provenance else {},
        token_weight=tuple(1.0 for _ in score.target_token_ids),
        student_logprobs=tuple(score.student_logprobs),
        teacher_logprobs=tuple(score.teacher_logprobs),
    )
