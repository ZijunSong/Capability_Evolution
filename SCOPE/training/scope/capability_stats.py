"""Capability-level statistics: U_c, δ_t, ρ_c (P7)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from harness.capability.capability_id import CapabilityId, parse_capability_id
from training.scope.schema import DecisionSupervisionSampleV3, Route


@dataclass
class CapabilityStats:
    capability_id: str
    n_proposed: int = 0
    n_verified_targets: int = 0
    n_endorse: int = 0
    n_correct: int = 0
    n_ignore: int = 0
    n_train: int = 0
    sum_local_gain: float = 0.0
    # Internalization: student already matches verified target
    n_student_matches_target: int = 0

    @property
    def reliability_u(self) -> float:
        """U_c = verified targets / proposed targets."""
        if self.n_proposed == 0:
            return 0.0
        return self.n_verified_targets / self.n_proposed

    @property
    def mean_local_gain(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.sum_local_gain / self.n_proposed

    @property
    def internalization_rho(self) -> float:
        """ρ_c ≈ fraction of verified cases where student already matches target."""
        denom = self.n_endorse + self.n_correct
        if denom == 0:
            return 0.0
        return self.n_student_matches_target / denom

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "n_proposed": self.n_proposed,
            "n_verified_targets": self.n_verified_targets,
            "n_endorse": self.n_endorse,
            "n_correct": self.n_correct,
            "n_ignore": self.n_ignore,
            "n_train": self.n_train,
            "reliability_u": self.reliability_u,
            "mean_local_gain": self.mean_local_gain,
            "internalization_rho": self.internalization_rho,
        }


@dataclass
class CapabilityStatsAggregator:
    stats: dict[str, CapabilityStats] = field(default_factory=dict)

    def _get(self, cap: str) -> CapabilityStats:
        if cap not in self.stats:
            self.stats[cap] = CapabilityStats(capability_id=cap)
        return self.stats[cap]

    def update(self, sample: DecisionSupervisionSampleV3) -> None:
        cap = sample.capability_id or "unknown"
        s = self._get(cap)
        s.n_proposed += 1
        if sample.route == Route.ENDORSE:
            s.n_endorse += 1
            s.n_verified_targets += 1
            s.n_student_matches_target += 1
        elif sample.route == Route.CORRECT:
            s.n_correct += 1
            s.n_verified_targets += 1
            if sample.student_action == sample.target_action:
                s.n_student_matches_target += 1
        else:
            s.n_ignore += 1
        if sample.train_mask and sample.route != Route.IGNORE:
            s.n_train += 1
        s.sum_local_gain += float(sample.weight_terms.local_gain)

    def update_many(self, samples: Iterable[DecisionSupervisionSampleV3]) -> None:
        for s in samples:
            self.update(s)

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.stats.items()}
