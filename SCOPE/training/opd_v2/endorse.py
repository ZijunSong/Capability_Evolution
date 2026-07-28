"""Endorse loss: teacher-advantage gated NLL on student action tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class EndorseLossOutput:
    loss: float
    gate: float
    gap: float
    n_tokens: int


def compute_endorse_loss(
    student_logps: Sequence[float],
    teacher_logps: Sequence[float],
    *,
    beta: float = 5.0,
    validity_mask: int = 1,
    module_weight: float = 1.0,
) -> EndorseLossOutput:
    """L_endorse = -gate.detach() * mean(logp_student_action) * validity * weight.

    Teacher branch and gate are stop-gradient (scalars here are already detached).
    """
    if not validity_mask or not student_logps:
        return EndorseLossOutput(loss=0.0, gate=0.0, gap=0.0, n_tokens=0)

    n = min(len(student_logps), len(teacher_logps)) if teacher_logps else len(student_logps)
    if n == 0:
        return EndorseLossOutput(loss=0.0, gate=0.0, gap=0.0, n_tokens=0)

    s_mean = sum(student_logps[:n]) / n
    if teacher_logps:
        t_mean = sum(teacher_logps[:n]) / n
        gap = t_mean - s_mean  # already treated as detached
    else:
        gap = 0.0
    gate = _sigmoid(beta * gap)
    # stop-grad on gate: use float value only
    gate_detached = float(gate)
    loss = -gate_detached * s_mean * float(module_weight)
    return EndorseLossOutput(
        loss=float(loss),
        gate=gate_detached,
        gap=float(gap),
        n_tokens=n,
    )


def compute_endorse_loss_batch(
    batch: list[dict],
    *,
    beta: float = 5.0,
) -> dict[str, float]:
    """Batch of dicts with student_logps, teacher_logps, validity_mask, module_weight, module_id."""
    total = 0.0
    n_valid = 0
    by_module: dict[str, list[float]] = {}
    gates: list[float] = []
    for item in batch:
        out = compute_endorse_loss(
            item.get("student_logps", []),
            item.get("teacher_logps", []),
            beta=beta,
            validity_mask=int(item.get("validity_mask", 1)),
            module_weight=float(item.get("module_weight", 1.0)),
        )
        if out.n_tokens == 0:
            continue
        total += out.loss
        n_valid += 1
        gates.append(out.gate)
        mid = str(item.get("module_id", "unknown"))
        by_module.setdefault(mid, []).append(out.loss)
    avg = total / max(1, n_valid)
    metrics = {
        "endorse_loss": avg,
        "endorse_n": float(n_valid),
        "endorse_gate_mean": sum(gates) / max(1, len(gates)),
    }
    for mid, losses in by_module.items():
        metrics[f"endorse_loss/{mid}"] = sum(losses) / max(1, len(losses))
    return metrics
