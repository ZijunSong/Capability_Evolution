"""Module lifecycle states."""

from __future__ import annotations

from enum import Enum


class LifecycleState(str, Enum):
    ACTIVE = "active"
    DISTILLING = "distilling"
    CONDITIONAL = "conditional"
    RETIRED = "retired"
    REACTIVATED = "reactivated"
