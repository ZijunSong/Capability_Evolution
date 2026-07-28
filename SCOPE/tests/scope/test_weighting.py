"""Module weighting tests."""

from __future__ import annotations

from training.opd_v2.weighting import ModuleWeightTracker, WeightingConfig


def test_fixed_weight():
    t = ModuleWeightTracker(WeightingConfig(enabled=False, lambda_0=0.01))
    assert t.lambda_for("verification") == 0.01


def test_reliability_weight():
    t = ModuleWeightTracker(
        WeightingConfig(enabled=True, mode="reliability", lambda_0=0.01, min_scale=0.1)
    )
    t.record_shadow("verification", mode="correct", valid=True, candidate_generated=True, candidate_valid=True)
    t.record_shadow("verification", mode="correct", valid=True, candidate_generated=True, candidate_valid=False)
    lam = t.lambda_for("verification")
    assert 0.001 <= lam <= 0.01
