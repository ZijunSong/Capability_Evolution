"""Statistics modes.

``pilot_reused_dev``: development panels may repeat across rounds; every
interval is a development diagnostic and must not be claimed as an independent
per-round significance result.  Final generalization comes from the one-shot
``final_test``.

``pre_registered_confirm``: rounds, reserve size, max attempts and alpha
allocation are fixed before the campaign; consumption of each confirm panel is
recorded exactly once (see campaign.splits).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatisticsPlan:
    mode: str  # pilot_reused_dev | pre_registered_confirm
    max_rounds: int
    confirm_attempts_per_round: int = 1
    alpha: float = 0.05
    alpha_spending: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("pilot_reused_dev", "pre_registered_confirm"):
            raise ValueError(f"unknown statistics mode {self.mode!r}")
        if self.mode == "pre_registered_confirm" and not self.alpha_spending:
            raise ValueError("pre_registered_confirm requires explicit alpha_spending")

    def is_development_diagnostic(self) -> bool:
        return self.mode == "pilot_reused_dev"

    def alpha_for_round(self, round_index: int) -> float:
        if not self.alpha_spending:
            return self.alpha
        idx = min(max(round_index, 0), len(self.alpha_spending) - 1)
        return float(self.alpha_spending[idx])
