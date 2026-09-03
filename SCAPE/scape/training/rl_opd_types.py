"""Hybrid RL + SR-OPD data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from scape.state.snapshot import EnvironmentSnapshot


TRAINING_MODE_RL = "rl"
TRAINING_MODE_PURE_OPD = "pure_opd"
TRAINING_MODE_RL_OPD = "rl_opd"
TRAINING_MODE_SCAPE_RL = "scape_rl"
TRAINING_MODE_SCAPE_SEED = "scape_seed"

# k < 0 means every structurally valid action on the trajectory.
OPD_STATES_ALL = -1
OPD_LOSS_CE = "sr_opd_ce"
# Legacy name kept as an alias of the SEED sampled-gap contract.
OPD_LOSS_REVERSE_KL = "sr_opd_reverse_kl"
OPD_LOSS_SAMPLED_GAP = "sr_opd_sampled_gap"
# SEED gated gap on *projected* student-legal actions (scape+seed).
OPD_LOSS_PROJECTED_GAP = "sr_opd_projected_gap"
SAMPLED_OPD_LOSSES = frozenset({OPD_LOSS_SAMPLED_GAP, OPD_LOSS_REVERSE_KL})
SEED_GAP_LOSSES = frozenset({OPD_LOSS_SAMPLED_GAP, OPD_LOSS_REVERSE_KL, OPD_LOSS_PROJECTED_GAP})
SCAPE_RL_LAMBDA_OPD = 0.01
SCAPE_RL_OPD_GATE_BETA = 5.0


def uses_sampled_opd(opd_loss: str) -> bool:
    """True for scape+rl: SEED gap on CISPO sampled tokens, no projector."""
    return str(opd_loss) in SAMPLED_OPD_LOSSES


def uses_seed_gap(opd_loss: str) -> bool:
    """Gated token-mean OPD (sampled or projected targets)."""
    return str(opd_loss) in SEED_GAP_LOSSES


def uses_projected_seed(opd_loss: str) -> bool:
    """True for scape+seed: projector + SEED-scale gap on a*."""
    return str(opd_loss) == OPD_LOSS_PROJECTED_GAP

UPDATE_RL_OPD_JOINT = "rl_opd_joint"
UPDATE_RL_ONLY = "rl_only"
UPDATE_OPD_ONLY_ZERO_RL = "opd_only_zero_rl_signal"
UPDATE_SKIPPED = "skipped_empty"

PROTOCOL_COMPLETE_RL_OPD = "sr_projected_rl_opd"
PROTOCOL_LEGACY_RL_PLUS_TOOL_KL = "legacy_rl_plus_tool_kl"

OPD_WEIGHT_NORMALIZATION = "per_optimizer_substep_token_mean"


@dataclass
class StudentDecisionPoint:
    episode_id: str
    query_id: str
    rollout_idx: int
    turn_id: int
    policy_version: str
    pre_action_snapshot: EnvironmentSnapshot
    pre_action_snapshot_hash: str
    student_model_input: Any
    student_action_tokens: list[int]
    student_action_text: str
    action_tool_names: list[str]
    post_action_snapshot: EnvironmentSnapshot | None = None
    reward: float | None = None
    structurally_valid: bool = True
    decision_point_id: str = ""
    student_prompt_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.decision_point_id:
            self.decision_point_id = f"{self.episode_id}:{self.turn_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "query_id": self.query_id,
            "rollout_idx": self.rollout_idx,
            "turn_id": self.turn_id,
            "policy_version": self.policy_version,
            "pre_action_snapshot": self.pre_action_snapshot.to_dict(),
            "pre_action_snapshot_hash": self.pre_action_snapshot_hash,
            "student_action_text": self.student_action_text,
            "action_tool_names": list(self.action_tool_names),
            "reward": self.reward,
            "structurally_valid": self.structurally_valid,
            "decision_point_id": self.decision_point_id,
        }


@dataclass
class HybridRolloutGroup:
    query_id: str
    policy_version: str
    trajectory_group: Any
    decision_points: list[StudentDecisionPoint]
    terminal_rewards: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridTrainingBatch:
    policy_version: str
    rl_datums: list[Any]
    opd_datums: list[Any]
    n_rl_tokens: int
    n_opd_tokens: int
    projection_stats: dict[str, Any] = field(default_factory=dict)
    reward_stats: dict[str, Any] = field(default_factory=dict)
    skipped_teacher: bool = False
    update_type: str = UPDATE_RL_OPD_JOINT
    opd_loss: str = "sr_opd_ce"


@dataclass
class HybridStepMetrics:
    update_type: str
    n_rl_datums: int
    n_opd_datums: int
    n_rl_tokens: int
    n_opd_tokens: int
    rl_loss_proxy: float | None
    opd_nll: float | None
    lambda_opd: float
    projection_coverage: float
    reject_rate: float
    policy_version: str
    n_rl_forward_backward: int = 0
    n_opd_forward_backward: int = 0
    n_optimizer_steps: int = 0
    opd_to_rl_token_ratio: float = 0.0
    rl_opd_exact_target_overlap_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
