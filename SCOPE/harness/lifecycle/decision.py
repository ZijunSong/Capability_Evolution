"""Lifecycle decision engine."""

from __future__ import annotations

from harness.lifecycle.contribution import ModuleAudit
from harness.lifecycle.state import LifecycleState


def decide(audit: ModuleAudit) -> LifecycleState:
    if audit.ci_after[1] < 0:
        return LifecycleState.RETIRED

    if audit.ci_after[0] <= 0 <= audit.ci_after[1]:
        return LifecycleState.RETIRED

    relative_drop = audit.relative_drop
    if relative_drop >= 0.7:
        return LifecycleState.CONDITIONAL

    return LifecycleState.ACTIVE
