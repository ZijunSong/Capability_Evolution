"""Paired comparison and retention statistics.

Relative performance must compare the same tasks under the same environment /
decoding seeds and the same budget protocol.  Multi-seed results are aggregated
to the task/cluster level before interval estimation.  Bootstrap resamples
task groups (clusters), not individual rollouts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PairedDelta:
    domain: str
    n_pairs: int
    mean_delta: float
    ci_low: float
    ci_high: float
    cluster_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "n_pairs": self.n_pairs,
            "mean_delta": self.mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "cluster_count": self.cluster_count,
        }


def paired_deltas(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    groups: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
    n_boot: int = 500,
) -> list[PairedDelta]:
    """Paired per-task deltas with a cluster bootstrap CI.

    ``baseline``/``candidate`` map task_id -> score on [0,1].  Tasks present in
    only one side are an error (panels must be complete and comparable).
    """
    missing = set(baseline) ^ set(candidate)
    if missing:
        raise ValueError(f"unpaired task ids: {sorted(missing)[:10]} ...")
    task_ids = sorted(baseline)
    if not task_ids:
        return []
    if groups is None:
        groups = {t: (t,) for t in task_ids}
    # Invert to cluster -> tasks (only tasks in the panel).
    clusters: dict[str, list[str]] = {}
    for gid, members in groups.items():
        present = [t for t in members if t in baseline]
        if present:
            clusters[gid] = present
    if not clusters:
        clusters = {t: [t] for t in task_ids}

    deltas = {t: candidate[t] - baseline[t] for t in task_ids}
    rng = random.Random(seed)
    cluster_ids = sorted(clusters)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample_vals: list[float] = []
        for _c in cluster_ids:
            gid = cluster_ids[rng.randrange(len(cluster_ids))]
            sample_vals.extend(deltas[t] for t in clusters[gid])
        boot_means.append(mean(sample_vals) if sample_vals else 0.0)
    boot_means.sort()
    lo = boot_means[int(0.025 * (n_boot - 1))]
    hi = boot_means[int(0.975 * (n_boot - 1))]
    return [
        PairedDelta(
            domain="suite",
            n_pairs=len(task_ids),
            mean_delta=mean(deltas.values()),
            ci_low=lo,
            ci_high=hi,
            cluster_count=len(cluster_ids),
        )
    ]


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 0, n_boot: int = 500, alpha: float = 0.05
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * (n_boot - 1))]
    hi = means[int((1 - alpha / 2) * (n_boot - 1))]
    return (lo, hi)
