"""Tests for stop calibration quadrant classification."""

from __future__ import annotations

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.stop_calibration import (
    StopQuadrantStats,
    classify_stop_quadrant,
)


def test_quadrant_stop_stop():
    student = CapabilityAction(CapabilityActionType.STOP_AND_ANSWER, {})
    shadow = CapabilityAction(CapabilityActionType.STOP_AND_ANSWER, {})
    assert classify_stop_quadrant(student, shadow) == "STOP→STOP"


def test_quadrant_continue_stop():
    student = CapabilityAction(CapabilityActionType.SEARCH, {"query": "x"})
    shadow = CapabilityAction(CapabilityActionType.STOP_AND_ANSWER, {})
    assert classify_stop_quadrant(student, shadow) == "CONTINUE→STOP"


def test_quadrant_stats_accumulate():
    stats = StopQuadrantStats()
    stats.record("STOP→CONTINUE")
    stats.record("CONTINUE→STOP")
    d = stats.to_dict()
    assert d["n_decision_points"] == 2
    assert d["STOP→CONTINUE"] == 1
    assert d["CONTINUE→STOP"] == 1
