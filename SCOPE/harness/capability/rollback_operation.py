"""Typed rollback recovery operations (Round 8 hard-control contract)."""

from __future__ import annotations

from enum import Enum


class RollbackOperation(str, Enum):
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    ROLLBACK_TO = "ROLLBACK_TO"


class RollbackReasonCode(str, Enum):
    NO_PROGRESS = "NO_PROGRESS"
    QUERY_LOOP = "QUERY_LOOP"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    TOOL_FAILURE = "TOOL_FAILURE"
    OFF_TRACK = "OFF_TRACK"
    BUDGET_TRAP = "BUDGET_TRAP"
    NONE = "NONE"
