from __future__ import annotations

from .forward_kl import forward_kl_exact


def next_turn_kl(student_logits, teacher_logits, mask=None):
    return forward_kl_exact(student_logits, teacher_logits, mask)
