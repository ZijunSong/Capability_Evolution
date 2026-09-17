"""Per-decision training records for the verl collector (T3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransitionRecord:
    run_id: str
    attempt_id: str
    rollout_batch_id: int
    query_id: str
    episode_id: str
    turn_id: int
    policy_version: str
    effective_prompt_ids: tuple[int, ...]
    prompt_hash: str
    response_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    response_loss_mask: tuple[int, ...]
    sampling_params: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    parse_valid: bool = False
    exec_ok: bool = False
    visible_doc_ids: tuple[str, ...] = ()
    terminal_reward: float | None = None
    reward_parts: dict[str, Any] = field(default_factory=dict)
    episode_advantage: float | None = None

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.behavior_logprobs):
            raise ValueError("response_ids/logprobs length mismatch")
        if len(self.response_ids) != len(self.response_loss_mask):
            raise ValueError("response_ids/mask length mismatch")
        if self.response_ids and not self.effective_prompt_ids:
            raise ValueError("missing effective_prompt_ids")
