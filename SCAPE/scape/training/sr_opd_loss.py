"""Unified Student-Realizable OPD cross-entropy.

This is the only formal SR-OPD objective. Component identity must not
appear here. The returned tensor is the same object that callers backward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


SR_OPD_LOSS_NAME = "sr_opd_ce"


def compute_sr_opd_ce(
    logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked, weighted token CE.

    L = - sum_m w_m sum_{k in M_m} log pi(a*_k | xi^S, a*_<k) / sum_m w_m |M_m|
    """
    if logprobs.numel() == 0:
        return logprobs.sum() * 0.0
    mask = token_mask.to(device=logprobs.device, dtype=logprobs.dtype)
    if token_weight is None:
        weight = mask
    else:
        weight = mask * token_weight.to(device=logprobs.device, dtype=logprobs.dtype)
    denom = weight.sum().clamp_min(1e-8)
    return -(logprobs * weight).sum() / denom


def sr_opd_ce_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE from raw logits. ``logits`` must require grad when training."""
    logprobs = F.log_softmax(logits, dim=-1)
    gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return compute_sr_opd_ce(gathered, token_mask, token_weight)


def pack_sr_opd_metrics(loss: torch.Tensor, *, n_supervised: float, weight: float = 1.0) -> dict[str, Any]:
    """Logger metrics derived from the same backward tensor."""
    return {
        "loss": float(loss.detach().item()),
        "sr_opd_ce": float(loss.detach().item()),
        "n_supervised_tokens": float(n_supervised),
        "weight": float(weight),
        "loss_impl": f"scape.training.sr_opd_loss:{SR_OPD_LOSS_NAME}",
        "loss_id": id(loss),
    }
