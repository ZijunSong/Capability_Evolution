"""Tests for lifecycle decision boundaries."""

from __future__ import annotations

from harness.lifecycle.contribution import ModuleAudit
from harness.lifecycle.decision import decide
from harness.lifecycle.distillability import compute_distillability
from harness.lifecycle.state import LifecycleState


def test_distillability_guard():
    assert compute_distillability(0.005, 0.0) is None


def test_retire_when_ci_crosses_zero():
    audit = ModuleAudit(
        module_id="verification",
        delta_before=0.2,
        delta_after=0.01,
        ci_before=(0.15, 0.25),
        ci_after=(-0.02, 0.04),
    )
    assert decide(audit) == LifecycleState.RETIRED


def test_conditional_on_large_relative_drop():
    audit = ModuleAudit(
        module_id="verification",
        delta_before=0.2,
        delta_after=0.04,
        ci_before=(0.15, 0.25),
        ci_after=(0.02, 0.06),
    )
    assert decide(audit) == LifecycleState.CONDITIONAL


def test_active_when_small_drop():
    audit = ModuleAudit(
        module_id="verification",
        delta_before=0.2,
        delta_after=0.15,
        ci_before=(0.15, 0.25),
        ci_after=(0.10, 0.20),
    )
    assert decide(audit) == LifecycleState.ACTIVE
