"""OPD loss functions."""

from __future__ import annotations

import math
from typing import Any


def compute_sampled_nll_loss(
    student_logps: list[float],
    weights: list[float],
) -> float:
    total_w = sum(weights) or 1.0
    nll = 0.0
    for lp, w in zip(student_logps, weights):
        if w <= 0:
            continue
        nll -= w * lp
    return nll / total_w


def compute_opd_loss(
    student_logps: list[float],
    teacher_logps: list[float],
    weights: list[float],
) -> float:
    """Token-level reverse KL proxy using scored log-probs."""
    total_w = sum(weights) or 1.0
    kl = 0.0
    for s_lp, t_lp, w in zip(student_logps, teacher_logps, weights):
        if w <= 0:
            continue
        kl += w * (math.exp(s_lp) * (s_lp - t_lp) if s_lp > -20 else (s_lp - t_lp))
    return kl / total_w


def combine_losses(
    opd_loss: float,
    rl_loss: float = 0.0,
    lambda_opd: float = 1.0,
) -> float:
    return rl_loss + lambda_opd * opd_loss
