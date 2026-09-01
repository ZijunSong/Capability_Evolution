"""Student-realizable OPD losses.

``sr_opd_ce`` is teacher-forced CE on the projected action given the reduced
prefix (rl+opd / pure OPD). ``sr_opd_sampled_gap`` is the SEED gated gap on
the same on-policy sampled tokens as CISPO (scape+rl). Component identity
must not appear here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from scape.training.canonical_metrics import kl_from_logits


SR_OPD_LOSS_NAME = "sr_opd_ce"
SR_OPD_REVERSE_KL_NAME = "sr_opd_reverse_kl"
SR_OPD_SAMPLED_GAP_NAME = "sr_opd_sampled_gap"
DEFAULT_OPD_GATE_BETA = 5.0


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


def reverse_kl_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(π_S || π_T) at each aligned position. Teacher is detached."""
    return kl_from_logits(teacher_logits.detach(), student_logits, forward=False)


def compute_sr_opd_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked reverse KL on the aligned action span.

    Same reduction as ``compute_sr_opd_ce``: weighted mean over masked tokens.
    """
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    kl = reverse_kl_per_token(student_logits, teacher_logits)
    mask = token_mask.to(device=kl.device, dtype=kl.dtype)
    if token_weight is None:
        weight = mask
    else:
        weight = mask * token_weight.to(device=kl.device, dtype=kl.dtype)
    denom = weight.sum().clamp_min(1e-8)
    return (kl * weight).sum() / denom


def gated_sampled_gap_per_token(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
) -> torch.Tensor:
    """SEED per-token term ``g · (sg[ℓ^T] − ℓ^S)`` with ``ℓ = log π(a)``.

    ``g = σ(β · sg[ℓ^T − ℓ^S])``. Teacher logprobs are detached; the gradient
    is a gated NLL on the sampled action, not a full-vocab reverse KL.
    """
    teacher = teacher_logprobs.detach().to(
        device=student_logprobs.device, dtype=student_logprobs.dtype
    )
    student = student_logprobs
    n = min(teacher.numel(), student.numel())
    teacher = teacher.reshape(-1)[:n]
    student = student.reshape(-1)[:n]
    delta = (teacher - student).detach()
    gate = torch.sigmoid(delta * float(gate_beta))
    return gate * (teacher - student)


def compute_sr_opd_sampled_gap(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
) -> torch.Tensor:
    """Masked token-mean of the SEED gated sampled-token gap."""
    if student_logprobs.numel() == 0:
        return student_logprobs.sum() * 0.0
    gap = gated_sampled_gap_per_token(
        student_logprobs, teacher_logprobs, gate_beta=gate_beta
    )
    if token_mask is None:
        weight = torch.ones_like(gap)
    else:
        weight = token_mask.to(device=gap.device, dtype=gap.dtype).reshape(-1)[: gap.numel()]
        if weight.numel() != gap.numel():
            weight = torch.ones_like(gap)
    denom = weight.sum().clamp_min(1e-8)
    return (gap * weight).sum() / denom


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
