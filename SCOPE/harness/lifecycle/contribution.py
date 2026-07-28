"""Module contribution and audit metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModuleAudit:
    module_id: str
    delta_before: float
    delta_after: float
    ci_before: tuple[float, float]
    ci_after: tuple[float, float]
    cost_before: float = 0.0
    cost_after: float = 0.0
    harm_rate_after: float = 0.0
    ood_delta_after: float = 0.0

    @property
    def relative_drop(self) -> float:
        return (self.delta_before - self.delta_after) / max(abs(self.delta_before), 1e-6)
