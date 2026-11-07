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
    n_invalid = 0
    # PR-03 (guide §4.1): unified outcome semantics.  The old `value < 1.0`
    # rule disagreed with the sibling modules' `value < 1e-9` rule — the same
    # partial-credit record was both "failure" (here) and "success" (there).
    # Success = strict success label; INVALID measurements are counted and
    # excluded from mechanism categories.
    from rsi.evolution.outcomes import Outcome, classify_outcome

    for tid, rec in sorted(per_task.items()):
        try:
            outcome = classify_outcome(rec)
        except ValueError:
            outcome = Outcome.INVALID
        if outcome is Outcome.INVALID:
            n_invalid += 1
            continue
        if outcome is Outcome.SUCCESS:
            successes.append(tid)
            continue
        kind = str(rec.get("error_kind") or rec.get("terminated_reason") or "wrong_result")
        failures[kind] += 1
        examples.setdefault(kind, []).append(tid)
    regressions: list[str] = []
    ref = parent_scores if parent_scores is not None else baseline_scores
    if ref:
        for tid, rec in sorted(per_task.items()):
            try:
                outcome = classify_outcome(rec)
            except ValueError:
                continue  # invalid: never a regression datum
            prev = ref.get(tid)
            if prev is not None and outcome is not Outcome.SUCCESS and classify_outcome(
                {"value": prev, "success": True}
            ) is Outcome.SUCCESS:
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
        trigger_stats={**dict(trigger_total), "_invalid_measurements": n_invalid},
        cost_stats=cost_sums,
    )


def _mechanism_hint(kind: str) -> str:
    # PR-5 (audit §5.3): a termination label is NOT a mechanism.  budget_wall
    # previously auto-mapped to "compress context / finish earlier", which fed
    # the proposer a fixed strategy and yielded a family of cosmetic
    # end-of-episode guidance components (the Hk ContextCompressor outcome).
    # Hints now describe the OBSERVATION and demand trace evidence before any
    # mechanism claim; when the label is ambiguous the honest answer is
    # UNKNOWN_NEEDS_TRACE.
    return {
        "model_invalid_action": (
            "model emits unparsable/illegal actions; inspect trace fragments for "
            "whether the schema is unclear, the format differs, or the model is "
            "truncating — the mechanism differs in each case"
        ),
        "tool_error": (
            "tool misuse or bad arguments; inspect traces for missing parameters, "
            "wrong API choice, or unverifiable arguments before proposing a fix"
        ),
        "budget_wall": (
            "UNKNOWN_NEEDS_TRACE: episodes hit the budget wall — this label alone "
            "cannot distinguish a clock/measurement error, slow inference, an "
            "invalid-action loop, redundant re-reading, or a genuinely long task; "
            "a compression or early-finish component is NOT justified by this "
            "label without trace evidence of wasted decisions"
        ),
        "max_actions": (
            "UNKNOWN_NEEDS_TRACE: action budget exhausted — determine whether the "
            "decisions were productive (long task) or repetitive (loop/missing "
            "state tracking) before proposing a mechanism"
        ),
        "finish_blocked": (
            "finish gating active; verify which component blocks finishing and "
            "whether its evidence requirement is satisfiable"
        ),
        "wrong_result": (
            "task executed but produced a wrong artifact; inspect traces for the "
            "first divergent decision and whether a verification step would have "
            "caught it"
        ),
    }.get(kind, "inspect example traces for the mechanism")
