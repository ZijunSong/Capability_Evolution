"""Capability-aware module weighting (fixed → reliability → full adaptive)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModuleStats:
    module_id: str

    shadow_calls: int = 0
    endorse_count: int = 0
    correct_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0

    agreement_count: int = 0
    evaluation_count: int = 0
    generated_candidate_count: int = 0
    valid_candidate_count: int = 0

    ema_contribution: float = 0.0
    ema_reliability: float = 0.0
    ema_internalization: float = 0.0

    def contribution(self) -> float:
        if self.shadow_calls <= 0:
            return 0.0
        return self.valid_count / self.shadow_calls

    def reliability(self) -> float:
        if self.generated_candidate_count <= 0:
            return 0.0
        return self.valid_candidate_count / self.generated_candidate_count

    def internalization(self) -> float:
        if self.evaluation_count <= 0:
            return 0.0
        return self.agreement_count / self.evaluation_count

    def to_dict(self) -> dict:
        return {
            "module_id": self.module_id,
            "shadow_calls": self.shadow_calls,
            "endorse_count": self.endorse_count,
            "correct_count": self.correct_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "agreement_count": self.agreement_count,
            "evaluation_count": self.evaluation_count,
            "generated_candidate_count": self.generated_candidate_count,
            "valid_candidate_count": self.valid_candidate_count,
            "ema_contribution": self.ema_contribution,
            "ema_reliability": self.ema_reliability,
            "ema_internalization": self.ema_internalization,
            "G": self.contribution(),
            "U": self.reliability(),
            "rho": self.internalization(),
        }


@dataclass
class WeightingConfig:
    enabled: bool = False
    mode: str = "fixed"  # fixed | reliability | full
    ema_decay: float = 0.95
    min_scale: float = 0.1
    max_scale: float = 1.0
    update_every: int = 20
    lambda_0: float = 0.01


@dataclass
class ModuleWeightTracker:
    config: WeightingConfig = field(default_factory=WeightingConfig)
    stats: dict[str, ModuleStats] = field(default_factory=dict)
    _steps: int = 0

    def get_stats(self, module_id: str) -> ModuleStats:
        if module_id not in self.stats:
            self.stats[module_id] = ModuleStats(module_id=module_id)
        return self.stats[module_id]

    def record_shadow(
        self,
        module_id: str,
        *,
        mode: str,
        valid: bool,
        candidate_generated: bool = False,
        candidate_valid: bool = False,
    ) -> None:
        s = self.get_stats(module_id)
        s.shadow_calls += 1
        if valid:
            s.valid_count += 1
        else:
            s.invalid_count += 1
        if mode == "endorse":
            s.endorse_count += 1
        elif mode == "correct":
            s.correct_count += 1
        if candidate_generated:
            s.generated_candidate_count += 1
            if candidate_valid:
                s.valid_candidate_count += 1

    def record_internalization(self, module_id: str, agree: bool) -> None:
        s = self.get_stats(module_id)
        s.evaluation_count += 1
        if agree:
            s.agreement_count += 1

    def _clip(self, x: float) -> float:
        return max(self.config.min_scale, min(self.config.max_scale, x))

    def lambda_for(self, module_id: str) -> float:
        if not self.config.enabled or self.config.mode == "fixed":
            return self.config.lambda_0
        s = self.get_stats(module_id)
        decay = self.config.ema_decay
        g = s.contribution()
        u = s.reliability()
        rho = s.internalization()
        s.ema_contribution = decay * s.ema_contribution + (1 - decay) * g
        s.ema_reliability = decay * s.ema_reliability + (1 - decay) * u
        s.ema_internalization = decay * s.ema_internalization + (1 - decay) * rho

        if self.config.mode == "reliability":
            scale = self._clip(s.ema_reliability if s.ema_reliability > 0 else u)
            return self.config.lambda_0 * scale

        # full adaptive
        raw = s.ema_contribution * s.ema_reliability * (1.0 - s.ema_internalization)
        if s.shadow_calls < 3:
            raw = 1.0
        return self.config.lambda_0 * self._clip(raw)

    def step(self) -> None:
        self._steps += 1
