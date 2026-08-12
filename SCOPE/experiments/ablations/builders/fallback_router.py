"""A10: Module retirement / fallback router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class Decision(Protocol):
    pass


@dataclass
class RouterTelemetry:
    n_calls: int = 0
    module_calls: int = 0
    fallback_calls: int = 0
    invalid_actions: int = 0
    catastrophic_failures: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def module_call_rate(self) -> float:
        return self.module_calls / max(self.n_calls, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "module_call_rate": self.module_call_rate,
            "fallback_calls": self.fallback_calls,
            "invalid_actions": self.invalid_actions,
            "catastrophic_failures": self.catastrophic_failures,
            "mean_latency_ms": (
                sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
            ),
        }


@dataclass
class FallbackPolicy:
    name: str
    should_call: Callable[[Any], bool]


def make_policy(variant: str, *, confidence_threshold: float = 0.5, budget: float = 0.2) -> FallbackPolicy:
    if variant in ("a10_internalized_only",):
        return FallbackPolicy("never", lambda d: False)
    if variant in ("a10_full_harness", "a10_always_fallback"):
        return FallbackPolicy("always", lambda d: True)
    if variant == "a10_confidence_fallback":
        def _conf(d: Any) -> bool:
            conf = float(getattr(d, "confidence", None) or (d.get("confidence") if isinstance(d, dict) else 1.0) or 1.0)
            return conf < confidence_threshold
        return FallbackPolicy("confidence", _conf)
    if variant == "a10_random_fallback_budget_matched":
        import random

        rng = random.Random(0)

        def _rand(d: Any) -> bool:
            return rng.random() < budget

        return FallbackPolicy("random_budget", _rand)
    raise ValueError(f"unknown A10 variant: {variant}")


def route_decision(
    state: Any,
    *,
    internalized_policy: Callable[[Any], Any],
    runtime_module: Callable[[Any], Any],
    fallback_policy: FallbackPolicy,
    telemetry: RouterTelemetry,
    latency_ms: float = 0.0,
) -> Any:
    """
    decision = internalized_policy(state)
    if fallback_policy.should_call(decision):
        decision = runtime_module(state)
    """
    telemetry.n_calls += 1
    telemetry.latencies_ms.append(latency_ms)
    decision = internalized_policy(state)
    if fallback_policy.should_call(decision):
        telemetry.module_calls += 1
        telemetry.fallback_calls += 1
        decision = runtime_module(state)
    # validate
    op = None
    if isinstance(decision, dict):
        op = decision.get("operation") or decision.get("action")
    else:
        op = getattr(decision, "operation", None) or getattr(decision, "action", None)
    if op in (None, "", "INVALID"):
        telemetry.invalid_actions += 1
    if isinstance(decision, dict) and decision.get("catastrophic"):
        telemetry.catastrophic_failures += 1
    return decision
