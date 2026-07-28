"""Correct-mode pairwise loss on recommended vs original actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _logsigmoid(x: float) -> float:
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def mean_token_logprob(logps: Sequence[float]) -> float:
    if not logps:
        return 0.0
    return sum(logps) / len(logps)


@dataclass
class CorrectLossOutput:
    loss: float
    margin: float
    n_orig: int
    n_rec: int


def compute_correct_loss(
    student_logps_original: Sequence[float],
    student_logps_recommended: Sequence[float],
    *,
    margin_scale: float = 1.0,
    validity_mask: int = 1,
    module_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> CorrectLossOutput:
    """Length-normalized pairwise: L = -logsigmoid(scale * (mean_rec - mean_orig))."""
    if not validity_mask or not student_logps_original or not student_logps_recommended:
        return CorrectLossOutput(loss=0.0, margin=0.0, n_orig=0, n_rec=0)

    s_orig = mean_token_logprob(student_logps_original)
    s_rec = mean_token_logprob(student_logps_recommended)
    margin = margin_scale * (s_rec - s_orig)
    loss = -_logsigmoid(margin) * float(module_weight)
    if label_smoothing > 0:
        # Mild smoothing toward zero margin
        loss = (1.0 - label_smoothing) * loss
    return CorrectLossOutput(
        loss=float(loss),
        margin=float(margin),
        n_orig=len(student_logps_original),
        n_rec=len(student_logps_recommended),
    )


def compute_correct_loss_batch(
    batch: list[dict],
    *,
    margin_scale: float = 1.0,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    total = 0.0
    n_valid = 0
    margins: list[float] = []
    by_module: dict[str, list[float]] = {}
    for item in batch:
        out = compute_correct_loss(
            item.get("student_logps_original", []),
            item.get("student_logps_recommended", []),
            margin_scale=margin_scale,
            validity_mask=int(item.get("validity_mask", 1)),
            module_weight=float(item.get("module_weight", 1.0)),
            label_smoothing=label_smoothing,
        )
        if out.n_orig == 0:
            continue
        total += out.loss
        n_valid += 1
        margins.append(out.margin)
        mid = str(item.get("module_id", "unknown"))
        by_module.setdefault(mid, []).append(out.loss)
    metrics = {
        "correct_loss": total / max(1, n_valid),
        "correct_n": float(n_valid),
        "correct_margin_mean": sum(margins) / max(1, len(margins)),
    }
    for mid, losses in by_module.items():
        metrics[f"correct_loss/{mid}"] = sum(losses) / max(1, len(losses))
    return metrics
