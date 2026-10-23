"""Performance/cost constrained selection (task book §7.3).

1. Drop protocol failures, severe domain regressions and budget overruns.
2. Among candidates whose performance is within a pre-registered tolerance of
   the best, pick the lower-cost one.  Complexity is a *vector* (active
   components, extra prompt tokens, extra LLM/tool calls, latency) — never
   code line count.

Relative performance compares the same tasks, same environment/decoding seeds
and same budget protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rsi.contracts.common import JSONValue


@dataclass(frozen=True)
class CostVector:
    active_components: int = 0
    extra_prompt_tokens: float = 0.0
    extra_llm_calls: float = 0.0
    extra_tool_calls: float = 0.0
    latency_ms: float = 0.0

    def scalar_cost(self) -> float:
        """Pre-registered monotone aggregation (documented in the run report)."""
        return (
            10.0 * self.active_components
            + 0.001 * self.extra_prompt_tokens
            + 1.0 * self.extra_llm_calls
            + 0.1 * self.extra_tool_calls
            + 0.001 * self.latency_ms
        )

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "active_components": self.active_components,
            "extra_prompt_tokens": self.extra_prompt_tokens,
            "extra_llm_calls": self.extra_llm_calls,
            "extra_tool_calls": self.extra_tool_calls,
            "latency_ms": self.latency_ms,
            "scalar_cost": self.scalar_cost(),
        }


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    macro_score: float
    per_domain: Mapping[str, float]
    cost: CostVector
    protocol_ok: bool = True
    budget_ok: bool = True

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "macro_score": self.macro_score,
            "per_domain": dict(self.per_domain),
            "cost": self.cost.to_json(),
            "protocol_ok": self.protocol_ok,
            "budget_ok": self.budget_ok,
        }


@dataclass(frozen=True)
class SelectionResult:
    champion_id: str | None
    reason: str
    ordered: tuple[str, ...]
    tolerance: float

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "champion_id": self.champion_id,
            "reason": self.reason,
            "ordered": list(self.ordered),
            "tolerance": self.tolerance,
        }


def select_champion(
    candidates: Sequence[CandidateScore],
    *,
    baseline: CandidateScore | None = None,
    tolerance: float = 0.01,
    max_domain_regression: float = 0.05,
) -> SelectionResult:
    legal = [c for c in candidates if c.protocol_ok and c.budget_ok]
    dropped = [c.candidate_id for c in candidates if not (c.protocol_ok and c.budget_ok)]
    if not legal:
        return SelectionResult(
            champion_id=baseline.candidate_id if baseline else None,
            reason="NO_LEGAL_CANDIDATES" + (f"; dropped={dropped}" if dropped else ""),
            ordered=(),
            tolerance=tolerance,
        )
    # Domain regression filter vs baseline
    if baseline is not None:
        kept = []
        for c in legal:
            regressed = False
            for domain, val in c.per_domain.items():
                base_val = baseline.per_domain.get(domain)
                if base_val is not None and val < base_val - max_domain_regression:
                    regressed = True
                    break
            if not regressed:
                kept.append(c)
        if kept:
            legal = kept
    best = max(c.macro_score for c in legal)
    contenders = [c for c in legal if c.macro_score >= best - tolerance]
    contenders.sort(key=lambda c: (c.cost.scalar_cost(), -c.macro_score, c.candidate_id))
    ordered = tuple(c.candidate_id for c in sorted(legal, key=lambda c: (-c.macro_score, c.cost.scalar_cost())))
    winner = contenders[0]
    reason = (
        f"best_within_tolerance={tolerance}; winner cost={winner.cost.scalar_cost():.3f} "
        f"macro={winner.macro_score:.4f}"
    )
    return SelectionResult(
        champion_id=winner.candidate_id,
        reason=reason,
        ordered=ordered,
        tolerance=tolerance,
    )
