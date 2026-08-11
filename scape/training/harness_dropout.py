"""Component availability curriculum / annealing schedules."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scape.adapters.components import RUNTIME_ANCHORS, full_mask, minus_mask


@dataclass
class DropoutSchedule:
    """Annealing from full harness availability toward slim mask."""

    target_components: list[str]
    total_steps: int
    mode: str = "linear"  # linear | one_shot | random | coalition | guided
    seed: int = 42
    keep_runtime_anchors: bool = True

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def availability(self, step: int) -> float:
        """Probability that a target component remains available (1=full, 0=removed)."""
        if self.mode == "one_shot":
            return 0.0
        if self.total_steps <= 0:
            return 0.0
        t = min(1.0, max(0.0, float(step) / float(self.total_steps)))
        if self.mode in {"linear", "coalition", "guided"}:
            return 1.0 - t
        if self.mode == "random":
            return 0.5
        raise ValueError(f"unknown dropout mode: {self.mode}")

    def sample_mask(self, step: int, *, base: Mapping[str, bool] | None = None) -> dict[str, bool]:
        mask = dict(base or full_mask())
        p_keep = self.availability(step)
        for cid in self.target_components:
            if self.keep_runtime_anchors and cid in RUNTIME_ANCHORS:
                continue
            if self.mode == "one_shot":
                mask[cid] = False
            elif self.mode == "random":
                mask[cid] = self._rng.random() < 0.5
            else:
                mask[cid] = self._rng.random() < p_keep
        return mask

    def sequential_stage_mask(self, stage_idx: int) -> dict[str, bool]:
        """For sequential A->B annealing: disable first stage_idx+1 targets."""
        mask = full_mask()
        for cid in self.target_components[: stage_idx + 1]:
            mask = minus_mask(cid, mask)
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_components": list(self.target_components),
            "total_steps": self.total_steps,
            "mode": self.mode,
            "seed": self.seed,
            "keep_runtime_anchors": self.keep_runtime_anchors,
        }


def one_shot_slim(components: Sequence[str]) -> dict[str, bool]:
    mask = full_mask()
    for cid in components:
        mask[cid] = False
    return mask
