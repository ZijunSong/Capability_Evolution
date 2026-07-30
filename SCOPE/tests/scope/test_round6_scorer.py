"""Tests for Round 6 scorer parity and decision config."""

from __future__ import annotations

import json

from harness.capability.dup_operation import DupOperation
from training.scope.decision_config import DupDecisionConfig
from training.scope.dup_operation_runtime import DupOperationRuntimeConfig
from training.scope.dup_telemetry import DupTelemetryAggregator


def test_decision_config_threshold():
    cfg = DupDecisionConfig(threshold=0.5)
    assert cfg.predict_from_margin(0.6) == DupOperation.SKIP_DUPLICATE
    assert cfg.predict_from_margin(0.4) == DupOperation.KEEP_EVIDENCE


def test_decision_config_bias():
    cfg = DupDecisionConfig(threshold=0.0, decision_bias=0.2)
    assert cfg.effective_threshold() == 0.2
    assert cfg.predict_from_margin(0.15) == DupOperation.KEEP_EVIDENCE


def test_score_sign_convention():
    """margin = score_skip - score_keep; positive margin → SKIP."""
    cfg = DupDecisionConfig()
    assert cfg.predict_from_scores(0.0, 0.5) == DupOperation.SKIP_DUPLICATE
    assert cfg.predict_from_scores(0.5, 0.0) == DupOperation.KEEP_EVIDENCE


def test_telemetry_serialization():
    tel = DupTelemetryAggregator()
    tel.score_events.append({
        "score_keep": 0.1,
        "score_skip": 0.3,
        "margin_skip_minus_keep": 0.2,
        "threshold": 0.0,
        "predicted_operation": "SKIP_DUPLICATE",
    })
    blob = json.dumps(tel.score_events)
    parsed = json.loads(blob)
    assert parsed[0]["margin_skip_minus_keep"] == 0.2


def test_runtime_config_has_decision():
    cfg = DupOperationRuntimeConfig(decision_config=DupDecisionConfig(threshold=1.0))
    assert cfg.decision_config.threshold == 1.0
