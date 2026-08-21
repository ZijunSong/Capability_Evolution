from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_kl_exact(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """KL(P_teacher || P_student) over the full vocabulary."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    per_token = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    if mask is None:
        loss = per_token.mean()
        denom = float(per_token.numel())
    else:
        weights = mask.float()
        denom_t = weights.sum().clamp_min(1.0)
        loss = (per_token * weights).sum() / denom_t
        denom = float(denom_t.detach().item())
    return loss, {"loss": float(loss.detach().item()), "forward_kl": float(loss.detach().item()), "n_tokens": denom}
