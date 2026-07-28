"""OPD data types and split rollout / training backend interfaces.

Architecture (OPHSD / veRL style):
  Rollout:  vLLM (or SGLang) — on-policy generation only
  Training: HF Transformers (+ FSDP2) — log-prob, KL/OPD loss, backward
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OPDTransition:
    episode_id: str
    query_id: str
    turn_id: int
    student_input_ids: list[int]
    action_ids: list[int]
    action_mask: list[bool]
    teacher_input_ids: list[int]
    privileged_module_id: str
    reward: float = 0.0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """One vLLM/HF-generation step; used to build OPDTransition."""

    prompt_token_ids: list[int]
    action_token_ids: list[int]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RolloutBackend(ABC):
    """High-throughput on-policy generation (vLLM, SGLang, Tinker)."""

    @abstractmethod
    def rollout_chat(
        self,
        messages: list[dict[str, str]],
        sampling_config: dict[str, Any],
    ) -> RolloutResult:
        ...


class TrainBackend(ABC):
    """Student/teacher forward, sampled-token log-prob, and optimizer step."""

    @abstractmethod
    def score_tokens(
        self, prefix_ids: list[int], target_ids: list[int]
    ) -> list[float]:
        ...

    @abstractmethod
    def train_step(
        self, batch: list[OPDTransition], loss_config: dict[str, Any]
    ) -> dict[str, float]:
        ...


class MockRolloutBackend(RolloutBackend):
    def rollout_chat(
        self,
        messages: list[dict[str, str]],
        sampling_config: dict[str, Any],
    ) -> RolloutResult:
        text = '{"tool":"search_corpus"}'
        prompt_ids = [1, 2, 3]
        action_ids = [10, 11, 12]
        _ = (messages, sampling_config)
        return RolloutResult(
            prompt_token_ids=prompt_ids,
            action_token_ids=action_ids,
            text=text,
        )


class MockTrainBackend(TrainBackend):
    def score_tokens(self, prefix_ids: list[int], target_ids: list[int]) -> list[float]:
        return [-0.05 * (i + 1) for i, _ in enumerate(target_ids)]

    def train_step(
        self, batch: list[OPDTransition], loss_config: dict[str, Any]
    ) -> dict[str, float]:
        _ = loss_config
        return {"loss": 0.1 * len(batch), "batch_size": float(len(batch))}


# Backward-compatible aliases (deprecated — prefer RolloutBackend / TrainBackend).
PolicyBackend = TrainBackend
MockPolicyBackend = MockTrainBackend
