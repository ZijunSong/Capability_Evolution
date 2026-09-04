"""System Contribution: Full Harness vs Full - component m."""

from __future__ import annotations

from typing import Any, Mapping

from trim.eval.paired_bootstrap import paired_query_stats


CORE_RETRIEVAL_METRICS = (
    "curated_recall",
    "trajectory_recall",
    "final_answer_recall",
    "harness_reward",
)

COST_METRICS = (
    "tool_calls",
    "turns",
    "context_tokens",
    "latency_ms",
    "state_ops",
)


def contribution_delta(
    full_by_qid: Mapping[str, Mapping[str, Any]],
    minus_by_qid: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """delta = full - minus_m (positive => component helps)."""
    # paired_query_stats computes other - base; set base=minus, other=full
    stats = paired_query_stats(
        minus_by_qid,
        full_by_qid,
        metric=metric,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "component_metric": metric,
        "delta_definition": "full - minus_m",
        **stats,
    }


def contribution_report(
    component_id: str,
    full_by_qid: Mapping[str, Mapping[str, Any]],
    minus_by_qid: Mapping[str, Mapping[str, Any]],
    *,
    metrics: tuple[str, ...] = CORE_RETRIEVAL_METRICS + COST_METRICS,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    per_metric: dict[str, Any] = {}
    for m in metrics:
        # Skip metrics absent from both sides
        sample = next(iter(full_by_qid.values()), {})
        if m not in sample and m not in next(iter(minus_by_qid.values()), {}):
            continue
        per_metric[m] = contribution_delta(
            full_by_qid, minus_by_qid, metric=m, n_boot=n_boot, seed=seed
        )
    quality_positive = False
    for m in CORE_RETRIEVAL_METRICS:
        if m in per_metric and per_metric[m]["mean_delta"] > 0:
            quality_positive = True
            break
    return {
        "component_id": component_id,
        "metrics": per_metric,
        "quality_positive": quality_positive,
    }
