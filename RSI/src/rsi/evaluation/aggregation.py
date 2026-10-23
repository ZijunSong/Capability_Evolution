"""Score aggregation.

Scores are official per-domain metrics on [0, 1].  Cross-domain aggregation is
an **equal-domain macro mean** — never a task-count-weighted pool that lets a
large dataset drown a small domain.  Per-domain results are always preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping, Sequence

from rsi.contracts.observation import ScoreRecord


@dataclass(frozen=True)
class DomainResult:
    domain: str
    task_ids: tuple[str, ...]
    values: tuple[float, ...]
    successes: tuple[bool, ...]
    metric_name: str
    synthetic: bool
    per_task: Mapping[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def score(self) -> float:
        return mean(self.values) if self.values else 0.0

    @property
    def success_rate(self) -> float:
        return mean(self.successes) if self.successes else 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "n": self.n,
            "score": self.score,
            "success_rate": self.success_rate,
            "metric_name": self.metric_name,
            "synthetic": self.synthetic,
            "task_ids": list(self.task_ids),
            "per_task": dict(self.per_task),
        }


@dataclass(frozen=True)
class SuiteResult:
    domains: tuple[DomainResult, ...]
    aggregate: str = "equal_domain_macro"

    @property
    def macro_score(self) -> float:
        if not self.domains:
            return 0.0
        return mean(d.score for d in self.domains)

    @property
    def synthetic(self) -> bool:
        return all(d.synthetic for d in self.domains)

    def to_json(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate,
            "macro_score": self.macro_score,
            "synthetic": self.synthetic,
            "domains": [d.to_json() for d in self.domains],
        }


def aggregate_scores(
    records: Sequence[ScoreRecord],
    *,
    aggregate: str = "equal_domain_macro",
) -> SuiteResult:
    if aggregate != "equal_domain_macro":
        raise ValueError(f"unsupported aggregation {aggregate!r}; only equal_domain_macro is registered")
    by_domain: dict[str, list[ScoreRecord]] = {}
    for r in records:
        by_domain.setdefault(r.domain, []).append(r)
    domains: list[DomainResult] = []
    for domain in sorted(by_domain):
        rs = by_domain[domain]
        # Deterministic task order; duplicated task ids are a protocol error.
        ids = [r.task_id for r in rs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate task ids in domain {domain}: {sorted(ids)}")
        order = sorted(range(len(rs)), key=lambda i: rs[i].task_id)
        rs = [rs[i] for i in order]
        domains.append(
            DomainResult(
                domain=domain,
                task_ids=tuple(r.task_id for r in rs),
                values=tuple(float(r.value) for r in rs),
                successes=tuple(bool(r.success) for r in rs),
                metric_name=rs[0].metric_name,
                synthetic=all(r.synthetic for r in rs),
                per_task={r.task_id: float(r.value) for r in rs},
            )
        )
    return SuiteResult(domains=tuple(domains), aggregate=aggregate)
