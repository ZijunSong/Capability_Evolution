"""Round-1 uniform weighting; adaptive P/U/rho reserved for Round 2+."""

from __future__ import annotations

from dataclasses import dataclass

from harness.capability.capability_id import CapabilityId, parse_capability_id
from harness.capability.distillability import round1_purity
from training.scope.schema import WeightTerms


@dataclass
class WeightingConfig:
    mode: str = "uniform"  # uniform | adaptive
    min_weight: float = 0.1
    max_weight: float = 1.0


def compute_weight_terms(
    capability_id: str | CapabilityId,
    *,
    reliability: float = 1.0,
    internalization: float = 0.0,
    local_gain: float = 1.0,
    config: WeightingConfig | None = None,
) -> WeightTerms:
    cfg = config or WeightingConfig()
    cap = parse_capability_id(capability_id)
    purity = round1_purity(cap).purity_score
    if cfg.mode == "uniform":
        return WeightTerms(
            procedural_purity=purity,
            reliability=1.0,
            internalization=0.0,
            local_gain=1.0,
        )
    # Adaptive (Round 2+): w ∝ P * U * (1-ρ) * δ
    return WeightTerms(
        procedural_purity=purity,
        reliability=float(reliability),
        internalization=float(internalization),
        local_gain=float(local_gain),
    )


def sample_weight_from_terms(
    terms: WeightTerms,
    *,
    config: WeightingConfig | None = None,
) -> float:
    cfg = config or WeightingConfig()
    if cfg.mode == "uniform":
        return 1.0 if terms.procedural_purity > 0 else 0.0
    raw = (
        terms.procedural_purity
        * terms.reliability
        * max(0.0, 1.0 - terms.internalization)
        * terms.local_gain
    )
    return float(max(cfg.min_weight, min(cfg.max_weight, raw)))
