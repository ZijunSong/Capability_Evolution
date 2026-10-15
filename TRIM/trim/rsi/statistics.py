"""Question-cluster paired deltas. Rollouts inside one question are not extra n."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Any, Mapping, Sequence


INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def cluster_paired_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 20261015,
    alpha: float = 0.05,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Average seeds inside a question, then average questions inside a cluster.

    ``rows`` items need ``cluster_id``, ``query_id``, ``j_base``, ``j_other``.
    An optional ``seed`` distinguishes decode replicates. Missing partners are
    not dropped just to shrink the denominator.
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    missing = 0
    for row in rows:
        if row.get("j_base") is None or row.get("j_other") is None:
            missing += 1
            if require_complete:
                return _insufficient(0, missing, reason="missing_pair")
            continue
        key = (str(row["cluster_id"]), str(row["query_id"]))
        grouped[key].append(float(row["j_other"]) - float(row["j_base"]))
    if not grouped:
        return _insufficient(0, missing, reason="no_pairs")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for (cluster_id, _query_id), deltas in grouped.items():
        by_cluster[cluster_id].append(sum(deltas) / len(deltas))
    cluster_deltas = [sum(values) / len(values) for values in by_cluster.values()]
    n = len(cluster_deltas)
    mean = sum(cluster_deltas) / n
    n_nonzero = sum(1 for value in cluster_deltas if abs(value) > 1e-12)
    if n < 2 or n_nonzero == 0:
        return {
            "status": INSUFFICIENT,
            "n_clusters": n,
            "n_questions": len(grouped),
            "n_rows": len(rows),
            "n_missing": missing,
            "mean_delta": mean,
            "lcb": None,
            "interval": None,
            "reason": "degenerate_or_tiny_sample",
            "ci_method": "cluster_bootstrap",
            "seed": seed,
            "alpha": alpha,
            "inferential_status": "approximate_pilot",
        }
    interval, lcb = _bootstrap(cluster_deltas, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "status": "ok",
        "n_clusters": n,
        "n_questions": len(grouped),
        "n_rows": len(rows),
        "n_missing": missing,
        "mean_delta": mean,
        "lcb": lcb,
        "interval": interval,
        "reason": "",
        "ci_method": "cluster_bootstrap",
        "seed": seed,
        "alpha": alpha,
        "inferential_status": "approximate_pilot",
    }


def _bootstrap(
    values: Sequence[float], *, n_boot: int, seed: int, alpha: float
) -> tuple[tuple[float, float], float]:
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    last = len(means) - 1
    lo = means[int((alpha / 2) * last)]
    hi = means[int((1 - alpha / 2) * last)]
    lcb = means[int(alpha * last)]
    return (lo, hi), lcb


def _insufficient(n: int, missing: int, *, reason: str) -> dict[str, Any]:
    return {
        "status": INSUFFICIENT,
        "n_clusters": n,
        "n_questions": 0,
        "n_rows": 0,
        "n_missing": missing,
        "mean_delta": None,
        "lcb": None,
        "interval": None,
        "reason": reason,
        "ci_method": "cluster_bootstrap",
        "inferential_status": "approximate_pilot",
    }
