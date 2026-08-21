from __future__ import annotations

import torch
import torch.nn.functional as F


def alpha_jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor | None = None, *, alpha: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_p = student_logp.exp()
    teacher_p = teacher_logp.exp()
    mix_p = alpha * teacher_p + (1.0 - alpha) * student_p
    mix_logp = mix_p.clamp_min(torch.finfo(mix_p.dtype).tiny).log()
    teacher_kl = (teacher_p * (teacher_logp - mix_logp)).sum(dim=-1)
    student_kl = (student_p * (student_logp - mix_logp)).sum(dim=-1)
    per_token = alpha * teacher_kl + (1.0 - alpha) * student_kl
    if mask is None:
        loss = per_token.mean()
        denom = float(per_token.numel())
    else:
        weights = mask.float()
        denom_t = weights.sum().clamp_min(1.0)
        loss = (per_token * weights).sum() / denom_t
        denom = float(denom_t.detach().item())
    return loss, {"loss": float(loss.detach().item()), "jsd": float(loss.detach().item()), "alpha": alpha, "n_tokens": denom}
