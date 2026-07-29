"""Paired statistics for E0 distillability (bootstrap CI, win/loss/tie)."""

from __future__ import annotations

import random
from typing import Sequence


def paired_deltas(
    off_by_q: dict[str, float],
    other_by_q: dict[str, float],
) -> list[float]:
    qids = sorted(set(off_by_q) & set(other_by_q))
    return [other_by_q[q] - off_by_q[q] for q in qids]


def paired_win_loss_tie(
    off_by_q: dict[str, float],
    other_by_q: dict[str, float],
    *,
    eps: float = 1e-9,
) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for q in sorted(set(off_by_q) & set(other_by_q)):
        d = other_by_q[q] - off_by_q[q]
        if d > eps:
            wins += 1
        elif d < -eps:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return [means[lo_idx], means[hi_idx]]


def bootstrap_ratio_ci(
    delta_proc: Sequence[float],
    delta_full: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[list[float], str]:
    if not delta_proc or not delta_full or len(delta_proc) != len(delta_full):
        return [0.0, 0.0], "LOW_CONFIDENCE"
    rng = random.Random(seed)
    n = len(delta_proc)
    ratios: list[float] = []
    for _ in range(n_resamples):
        proc_vals = []
        full_vals = []
        for _ in range(n):
            i = rng.randrange(n)
            proc_vals.append(delta_proc[i])
            full_vals.append(delta_full[i])
        dp = sum(proc_vals) / n
        df = sum(full_vals) / n
        if abs(df) < 1e-12:
            continue
        ratios.append(dp / df)
    if len(ratios) < n_resamples // 4:
        return [0.0, 0.0], "LOW_CONFIDENCE"
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios)) - 1]
    width = hi - lo
    confidence = "LOW_CONFIDENCE" if width > 2.0 else "HIGH"
    return [lo, hi], confidence


def compute_distillability(
    *,
    metric: str,
    off_by_q: dict[str, float],
    proc_by_q: dict[str, float],
    full_by_q: dict[str, float],
    min_effect_size: float,
    n_resamples: int = 1000,
    seed: int = 42,
):
    from training.scope.distillability.schema import DistillabilityMetricResult

    qids = sorted(set(off_by_q) & set(proc_by_q) & set(full_by_q))
    n = len(qids)
    if n == 0:
        return DistillabilityMetricResult(
            metric=metric,
            R_off=0.0,
            R_proc=0.0,
            R_full=0.0,
            delta_proc=0.0,
            delta_full=0.0,
            P_raw=None,
            P_clipped=None,
            probe_valid=False,
            invalid_reason="no_overlap_queries",
            n_queries=0,
            confidence="LOW_CONFIDENCE",
        )

    R_off = sum(off_by_q[q] for q in qids) / n
    R_proc = sum(proc_by_q[q] for q in qids) / n
    R_full = sum(full_by_q[q] for q in qids) / n
    delta_proc = R_proc - R_off
    delta_full = R_full - R_off

    delta_proc_i = [proc_by_q[q] - off_by_q[q] for q in qids]
    delta_full_i = [full_by_q[q] - off_by_q[q] for q in qids]

    probe_valid = True
    invalid_reason = ""
    P_raw = None
    P_clipped = None
    confidence = "HIGH"

    if abs(delta_full) < min_effect_size:
        probe_valid = False
        invalid_reason = "full_effect_too_small"
    else:
        P_raw = delta_proc / delta_full
        P_clipped = max(0.0, min(1.0, P_raw))

    ci_proc = bootstrap_ci(delta_proc_i, n_resamples=n_resamples, seed=seed)
    ci_full = bootstrap_ci(delta_full_i, n_resamples=n_resamples, seed=seed)
    ci_p, conf = bootstrap_ratio_ci(
        delta_proc_i, delta_full_i, n_resamples=n_resamples, seed=seed
    )
    confidence = conf if probe_valid else "LOW_CONFIDENCE"

    w, l, t = paired_win_loss_tie(off_by_q, proc_by_q)
    return DistillabilityMetricResult(
        metric=metric,
        R_off=R_off,
        R_proc=R_proc,
        R_full=R_full,
        delta_proc=delta_proc,
        delta_full=delta_full,
        P_raw=P_raw,
        P_clipped=P_clipped,
        probe_valid=probe_valid,
        invalid_reason=invalid_reason,
        ci95={
            "delta_proc": ci_proc,
            "delta_full": ci_full,
            "P": ci_p,
        },
        paired_wins=w,
        paired_losses=l,
        paired_ties=t,
        confidence=confidence,
        n_queries=n,
    )
