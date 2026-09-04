"""Paired query-level statistics (adapted from SCOPE inference/scope/paired_stats).

Allowed migration: generic paired W/L/T + bootstrap CI. No SCOPE method logic.
"""

from __future__ import annotations

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
    """Compute paired mean delta (other - base), bootstrap CI, win/loss/tie."""
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


def pair_by_query_id(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    id_key: str = "query_id",
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    a = {str(r[id_key]): r for r in rows_a}
    b = {str(r[id_key]): r for r in rows_b}
    if len(a) != len(rows_a) or len(b) != len(rows_b):
        raise ValueError("duplicate query_id in input rows")
    return a, b
