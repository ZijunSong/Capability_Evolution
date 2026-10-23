"""Per-generation diagnosis from real execution feedback.

Diagnosis is refreshed every generation (never the initial H0 failure counts
forever).  Both successes and failures are sampled so the search does not only
chase hard tasks and break learned abilities.  Task ids appear as audit
references only.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsi.contracts.common import JSONValue, digest_of


@dataclass(frozen=True)
class FailureCategory:
    kind: str  # e.g. model_invalid_action | tool_error | budget_wall | wrong_result | finish_blocked
    count: int
    example_task_ids: tuple[str, ...] = ()
    mechanism_summary: str = ""


@dataclass(frozen=True)
class GenerationDiagnosis:
    generation: int
    failure_categories: tuple[FailureCategory, ...]
    success_sample_task_ids: tuple[str, ...]
    regression_task_ids: tuple[str, ...]
    trigger_stats: Mapping[str, int]
    cost_stats: Mapping[str, float]
    notes: str = ""

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "generation": self.generation,
            "failure_categories": [
                {
                    "kind": f.kind,
                    "count": f.count,
                    "example_task_ids": list(f.example_task_ids),
                    "mechanism_summary": f.mechanism_summary,
                }
                for f in self.failure_categories
            ],
            "success_sample_task_ids": list(self.success_sample_task_ids),
            "regression_task_ids": list(self.regression_task_ids),
            "trigger_stats": dict(self.trigger_stats),
            "cost_stats": dict(self.cost_stats),
            "notes": self.notes,
        }


def diagnose_traces(
    *,
    generation: int,
    per_task: Mapping[str, Mapping[str, Any]],
    baseline_scores: Mapping[str, float] | None = None,
    parent_scores: Mapping[str, float] | None = None,
    max_examples: int = 3,
) -> GenerationDiagnosis:
    """Build a fresh diagnosis from per-task evaluation records.

    ``per_task`` maps task_id -> record with keys ``value``, ``terminated_reason``,
    optionally ``error_kind``, ``n_decisions``, ``trigger_stats``.
    """
    failures: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    successes: list[str] = []
    for tid, rec in sorted(per_task.items()):
        value = float(rec.get("value", 0.0))
        if value >= 1.0:
            successes.append(tid)
            continue
        kind = str(rec.get("error_kind") or rec.get("terminated_reason") or "wrong_result")
        failures[kind] += 1
        examples.setdefault(kind, []).append(tid)
    regressions: list[str] = []
    ref = parent_scores if parent_scores is not None else baseline_scores
    if ref:
        for tid, rec in sorted(per_task.items()):
            prev = ref.get(tid)
            if prev is not None and float(prev) >= 1.0 and float(rec.get("value", 0.0)) < 1.0:
                regressions.append(tid)
    trigger_total: Counter[str] = Counter()
    cost_sums: dict[str, float] = {}
    for rec in per_task.values():
        for k, v in dict(rec.get("trigger_stats") or {}).items():
            trigger_total[k] += int(v)
        for k, v in dict(rec.get("cost_stats") or {}).items():
            cost_sums[k] = cost_sums.get(k, 0.0) + float(v)
    return GenerationDiagnosis(
        generation=generation,
        failure_categories=tuple(
            FailureCategory(
                kind=kind,
                count=count,
                example_task_ids=tuple(examples.get(kind, [])[:max_examples]),
                mechanism_summary=_mechanism_hint(kind),
            )
            for kind, count in failures.most_common()
        ),
        success_sample_task_ids=tuple(successes[:max_examples]),
        regression_task_ids=tuple(regressions[:max_examples]),
        trigger_stats=dict(trigger_total),
        cost_stats=cost_sums,
    )


def _mechanism_hint(kind: str) -> str:
    return {
        "model_invalid_action": "model emits unparsable/illegal actions; consider a proposal validator or format guidance component",
        "tool_error": "tool misuse or bad arguments; consider argument checking/retry components",
        "budget_wall": "episodes run out of budget; consider context compression or earlier finishing policy",
        "max_actions": "too many actions before finishing; consider progress summarization or termination heuristic",
        "finish_blocked": "finish gating active; verify the gating component",
        "wrong_result": "task executed but produced wrong artifact; consider verification-before-finish components",
    }.get(kind, "inspect example traces for the mechanism")
