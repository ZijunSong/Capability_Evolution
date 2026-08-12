"""Paired query-level statistics for matched evaluations."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        raise ValueError("empty values for bootstrap_ci")
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * (len(means) - 1))]
    hi = means[int((1 - alpha / 2) * (len(means) - 1))]
    return lo, hi


def paired_query_stats(
    base: Mapping[str, Mapping[str, Any] | float],
    other: Mapping[str, Mapping[str, Any] | float],
    *,
    metric: str = "recall",
    n_boot: int = 1000,
    seed: int = 42,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Compute paired mean delta, bootstrap CI, win/loss/tie for matched queries."""
    base_ids = set(base)
    other_ids = set(other)
    missing_in_other = sorted(base_ids - other_ids)
    missing_in_base = sorted(other_ids - base_ids)
    if require_complete and (missing_in_other or missing_in_base):
        raise ValueError(
            f"missing query ids — base_only={missing_in_other[:5]} "
            f"other_only={missing_in_base[:5]}"
        )
    shared = sorted(base_ids & other_ids)
    if not shared:
        raise ValueError("no shared query ids for paired stats")

    def _get(row: Mapping[str, Any] | float) -> float:
        if isinstance(row, (int, float)):
            return float(row)
        if metric not in row:
            raise KeyError(f"metric {metric} missing for query row")
        return float(row[metric])

    deltas: list[float] = []
    wins = losses = ties = 0
    for qid in shared:
        b = _get(base[qid])
        o = _get(other[qid])
        d = o - b
        deltas.append(d)
        if d > 1e-6:
            wins += 1
        elif d < -1e-6:
            losses += 1
        else:
            ties += 1

    lo, hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    mean_delta = sum(deltas) / len(deltas)
    return {
        "metric": metric,
        "n": len(shared),
        "mean_delta": mean_delta,
        "bootstrap_ci_95": [lo, hi],
        "win": wins,
        "loss": losses,
        "tie": ties,
        "missing_in_other": missing_in_other,
        "missing_in_base": missing_in_base,
        "deltas": deltas,
    }


def seed_mean_std(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("empty values")
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
        "n": len(values),
    }


def rollout_seed_variance(
    per_seed: Mapping[int | str, Sequence[float]],
) -> dict[str, Any]:
    """Variance of query-mean metric across rollout seeds."""
    if not per_seed:
        raise ValueError("empty per_seed")
    means = []
    for seed, vals in per_seed.items():
        if not vals:
            raise ValueError(f"empty values for seed {seed}")
        means.append(sum(vals) / len(vals))
    return {
        "seed_means": means,
        **seed_mean_std(means),
        "range": max(means) - min(means),
    }
