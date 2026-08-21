from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_action_ce(student_logits: torch.Tensor, target_token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    logp = F.log_softmax(student_logits.float(), dim=-1)
    gathered = logp.gather(-1, target_token_ids.long().unsqueeze(-1)).squeeze(-1)
    if mask is None:
        loss = -gathered.mean()
        denom = float(gathered.numel())
    else:
        weights = mask.float()
        denom_t = weights.sum().clamp_min(1.0)
        loss = -(gathered * weights).sum() / denom_t
        denom = float(denom_t.detach().item())
    return loss, {"loss": float(loss.detach().item()), "action_ce": float(loss.detach().item()), "n_tokens": denom}
