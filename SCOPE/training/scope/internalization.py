"""Internalization helpers (ρ_c) — Round 2+/3 retirement uses these."""

from __future__ import annotations

from training.scope.capability_stats import CapabilityStatsAggregator


def summarize_internalization(agg: CapabilityStatsAggregator) -> dict[str, float]:
    return {k: v.internalization_rho for k, v in agg.stats.items()}
