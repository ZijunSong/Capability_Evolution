"""A10 fallback router tests."""

from __future__ import annotations

from experiments.ablations.builders.fallback_router import (
    RouterTelemetry,
    make_policy,
    route_decision,
)


def test_internalized_only_never_calls_module():
    policy = make_policy("a10_internalized_only")
    tel = RouterTelemetry()
    d = route_decision(
        {"confidence": 0.1},
        internalized_policy=lambda s: {"operation": "KEEP_EVIDENCE", "confidence": 0.1},
        runtime_module=lambda s: {"operation": "SKIP_DUPLICATE"},
        fallback_policy=policy,
        telemetry=tel,
    )
    assert d["operation"] == "KEEP_EVIDENCE"
    assert tel.module_calls == 0


def test_always_fallback_calls_module():
    policy = make_policy("a10_always_fallback")
    tel = RouterTelemetry()
    d = route_decision(
        {},
        internalized_policy=lambda s: {"operation": "KEEP_EVIDENCE"},
        runtime_module=lambda s: {"operation": "SKIP_DUPLICATE", "from_module": True},
        fallback_policy=policy,
        telemetry=tel,
    )
    assert d.get("from_module") is True
    assert tel.module_call_rate == 1.0
