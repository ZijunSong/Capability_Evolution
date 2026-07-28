"""Distillability computation with stability guards."""

from __future__ import annotations

MIN_EFFECT = 0.01
EPS = 1e-6


def compute_distillability(delta_before: float, delta_after: float) -> float | None:
    if delta_before < MIN_EFFECT:
        return None
    return 1.0 - delta_after / max(delta_before, EPS)


def compute_module_delta(metric_full: float, metric_minus: float) -> float:
    return metric_full - metric_minus
