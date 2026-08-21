from __future__ import annotations

from .reverse_kl import reverse_kl_exact


def step_weighted_kl(student_logits, teacher_logits, mask=None, *, step_weights=None):
    loss, metrics = reverse_kl_exact(student_logits, teacher_logits, mask)
    metrics["step_weighted"] = True
    return loss, metrics
