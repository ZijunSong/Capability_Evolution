from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EffectType = Literal[
    "ACTION_SPACE_CHANGE",
    "ARGUMENT_PRIVILEGE",
    "AUTOMATIC_SIDE_EFFECT",
    "AUTOMATIC_POOL_FILTER",
    "EXTERNAL_INFORMATION_AUGMENTATION",
    "PRIVILEGED_CONTEXT",
    "PRIVILEGED_RETRIEVAL_CONTEXT",
]

Realizability = Literal["DIRECT", "PROJECTABLE", "PARTIAL", "NON_REALIZABLE"]
LossMode = Literal[
    "forward_kl",
    "reverse_kl",
    "jsd",
    "action_ce",
    "projected_action_ce",
    "next_turn_kl",
    "step_weighted_kl",
    "hybrid_rl_opd",
    "none",
]


@dataclass(frozen=True)
class ToolAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ComponentTransitionRecord:
    query_id: str
    trajectory_id: str
    turn_id: int
    component_name: str
    component_effect_type: str
    realizability: str
    student_checkpoint_sha: str = ""
    teacher_checkpoint_sha: str = ""
    state_hash_pre: str = ""
    student_view_hash: str = ""
    teacher_view_hash: str = ""
    student_prompt_token_ids: list[int] = field(default_factory=list)
    student_response_token_ids: list[int] = field(default_factory=list)
    response_mask: list[bool] = field(default_factory=list)
    tool_span_mask: list[bool] = field(default_factory=list)
    argument_span_mask: list[bool] = field(default_factory=list)
    student_logprobs: list[float] = field(default_factory=list)
    teacher_logprobs: list[float] = field(default_factory=list)
    reference_logprobs: list[float] = field(default_factory=list)
    teacher_route_distribution: dict[str, float] | None = None
    teacher_action: dict[str, Any] | None = None
    component_event_active: bool = False
    component_effect: dict[str, Any] = field(default_factory=dict)
    projected_action: dict[str, Any] | None = None
    projection_valid: bool = False
    visibility_valid: bool = False
    reward_before: float | None = None
    reward_after: float | None = None
    trajectory_reward: float | None = None
    terminal_reward: float | None = None
    visible_doc_ids: list[str] = field(default_factory=list)
    curated_ids_pre: list[str] = field(default_factory=list)
    curated_ids_post: list[str] = field(default_factory=list)
    query_split: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
