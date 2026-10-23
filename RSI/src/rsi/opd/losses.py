"""OPD / CISPO loss pure functions.

Migrated from the legacy ``sr_opd_loss`` implementation with behavior kept
numerically identical (see ``tests/unit/test_loss_and_grad_parity``; the legacy
reference is embedded in the test so this package stays standalone).

Semantics (task book §9.1):

- ``gated_action_gap_per_token`` implements the SEED gated log-probability gap
  ``g * (sg[lT] - lS)`` with ``g = sigmoid(beta * sg[lT - lS])``.  Teacher and
  gate are stop-gradient.  This is NOT a full-vocab reverse KL
  (``compute_sr_opd_reverse_kl`` is a separate path and must not be conflated).
- The OPD objective is

      L_OPD = sum_t w_t m_t g_t (sg[lT_t] - lS_t) / sum_t w_t m_t

  (``gated_projected_gap_loss``), and the joint objective is
  ``L = L_CISPO + lambda * L_OPD``.
- CISPO core: ``ratio = exp(new_lp - old_lp)``,
  ``weight = stopgrad(clamp(ratio, low, high))``,
  token term ``-weight * advantage * new_lp * mask``, reduced by the global
  effective-token count.  This must not silently become a PPO surrogate or a
  GRPO default objective.

No supervised tokens -> zero contribution plus a recorded reason; dummy targets
are never manufactured.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_OPD_GATE_BETA = 5.0
DEFAULT_CLIP_LOW = 0.8
DEFAULT_CLIP_HIGH = 1.2
GATED_PROJECTED_GAP_V1 = "gated_projected_gap_v1"


# ---------------------------------------------------------------------------
# masked weighted CE (legacy parity)
# ---------------------------------------------------------------------------


def compute_sr_opd_ce(
    logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked, weighted token CE: weighted mean of ``-logprob`` over mask."""
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
    logprobs = F.log_softmax(logits.float(), dim=-1)
    gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return compute_sr_opd_ce(gathered, token_mask, token_weight)


# ---------------------------------------------------------------------------
# reverse KL (separate legacy path; not the projected gap)
# ---------------------------------------------------------------------------


def kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    forward: bool = True,
) -> torch.Tensor:
    t_logp = F.log_softmax(teacher_logits, dim=-1)
    s_logp = F.log_softmax(student_logits, dim=-1)
    if forward:
        t_p = t_logp.exp()
        return (t_p * (t_logp - s_logp)).sum(dim=-1)
    s_p = s_logp.exp()
    return (s_p * (s_logp - t_logp)).sum(dim=-1)


def reverse_kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return kl_from_logits(teacher_logits.detach(), student_logits, forward=False)


def compute_sr_opd_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight: torch.Tensor | None = None,
) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# gated action gap (legacy parity)
# ---------------------------------------------------------------------------


def action_gap_gate(
    delta: torch.Tensor,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
    gate_mode: str = "sigmoid",
) -> torch.Tensor:
    """Reliability weight on a detached action-gap.

    ``sigmoid`` is ``sigma(beta * sg[lT - lS])``.  ``none`` is the constant-1
    tensor used by the ``no_gate`` ablation.  ``beta=0`` is NOT that switch: it
    yields ``g=0.5``.
    """
    mode = str(gate_mode or "sigmoid")
    if mode == "sigmoid":
        return torch.sigmoid(delta.detach() * float(gate_beta))
    if mode == "none":
        return torch.ones_like(delta)
    raise ValueError(f"unknown gate_mode={gate_mode!r}")


def gated_action_gap_per_token(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
    gate_mode: str = "sigmoid",
) -> torch.Tensor:
    """Per-token ``g * (sg[lT] - lS)``.

    Teacher logprobs detached; gradient is a gated NLL on the scored action.
    Length mismatch is an error; silent ``min()`` truncation is forbidden.
    """
    teacher = teacher_logprobs.detach().to(
        device=student_logprobs.device, dtype=student_logprobs.dtype
    ).reshape(-1)
    student = student_logprobs.reshape(-1)
    if teacher.numel() != student.numel():
        raise ValueError(
            f"gated action-gap length mismatch: teacher={teacher.numel()} student={student.numel()}"
        )
    delta = (teacher - student).detach()
    gate = action_gap_gate(delta, gate_beta=gate_beta, gate_mode=gate_mode)
    return gate * (teacher - student)


def gated_action_gap_weighted_sum(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    weights: torch.Tensor,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
    gate_mode: str = "sigmoid",
) -> torch.Tensor:
    gap = gated_action_gap_per_token(
        student_logprobs, teacher_logprobs, gate_beta=gate_beta, gate_mode=gate_mode
    )
    weight = weights.to(device=gap.device, dtype=gap.dtype).reshape(-1)
    if weight.numel() != gap.numel():
        raise ValueError(
            f"gated action-gap weight length mismatch: weights={weight.numel()} gap={gap.numel()}"
        )
    return (weight * gap).sum()


def scale_projected_gap_loss(
    numerator: torch.Tensor,
    *,
    lambda_opd: float,
    z_gap: float,
    world_size: int,
) -> torch.Tensor:
    """Undo DDP/FSDP mean reduction: ``world_size * lambda * numer / Z_gap``.

    ``Z_gap`` is the coordinator's global weight sum; do not all-reduce it.
    A zero denominator skips real gap grads without ``clamp_min(1)``.
    """
    if float(z_gap) <= 0.0:
        return numerator * 0.0
    return float(world_size) * float(lambda_opd) * numerator / float(z_gap)


def scale_gated_gap_loss(
    numerator: torch.Tensor,
    *,
    lambda_opd: float,
    z_gap: float,
    world_size: int,
) -> torch.Tensor:
    return scale_projected_gap_loss(
        numerator, lambda_opd=lambda_opd, z_gap=z_gap, world_size=world_size
    )


def compute_sr_opd_sampled_gap(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
    gate_mode: str = "sigmoid",
) -> torch.Tensor:
    if student_logprobs.numel() == 0:
        return student_logprobs.sum() * 0.0
    gap = gated_action_gap_per_token(
        student_logprobs, teacher_logprobs, gate_beta=gate_beta, gate_mode=gate_mode
    )
    if token_mask is None:
        weight = torch.ones_like(gap)
    else:
        weight = token_mask.to(device=gap.device, dtype=gap.dtype).reshape(-1)
        if weight.numel() != gap.numel():
            raise ValueError(
                f"sampled-gap mask length mismatch: mask={weight.numel()} gap={gap.numel()}"
            )
    denom = weight.sum().clamp_min(1e-8)
    return (gap * weight).sum() / denom


# ---------------------------------------------------------------------------
# the registered objective: gated_projected_gap_v1
# ---------------------------------------------------------------------------


def gated_projected_gap_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight: torch.Tensor | None = None,
    *,
    gate_beta: float = DEFAULT_OPD_GATE_BETA,
    gate_mode: str = "sigmoid",
    zero_denominator_policy: str = "skip",
) -> torch.Tensor:
    """L_OPD = sum_t w_t m_t g_t (sg[lT] - lS) / sum_t w_t m_t.

    Teacher and gate stop-gradient: the gradient is a weighted NLL toward the
    aligned target action, not a full-vocab reverse KL.  With zero supervised
    weight the loss contributes zero (``skip``) — no dummy targets.
    """
    gap = gated_action_gap_per_token(
        student_logprobs, teacher_logprobs, gate_beta=gate_beta, gate_mode=gate_mode
    )
    mask = token_mask.to(device=gap.device, dtype=gap.dtype).reshape(-1)
    if token_weight is None:
        weight = mask
    else:
        weight = mask * token_weight.to(device=gap.device, dtype=gap.dtype).reshape(-1)
    if weight.numel() != gap.numel():
        raise ValueError(f"mask/weight length mismatch: {weight.numel()} vs {gap.numel()}")
    denom = weight.sum()
    if float(denom) <= 0.0:
        if zero_denominator_policy == "skip":
            return gap.sum() * 0.0
        raise ValueError("gated_projected_gap_loss: zero denominator")
    return (weight * gap).sum() / denom


# ---------------------------------------------------------------------------
# CISPO core (legacy parity with hf_rl_opd_client)
# ---------------------------------------------------------------------------


def cispo_ratio(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
) -> torch.Tensor:
    return (new_logprobs - old_logprobs).exp()


def cispo_token_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: float | torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
    normalizer: float | None = None,
) -> torch.Tensor:
    """CISPO token objective: ``-stopgrad(clamp(ratio)) * adv * new_lp * mask``.

    Reduced by ``normalizer`` (global effective action tokens) when given,
    otherwise by the local masked-token count.
    """
    new_lp = new_logprobs.float()
    old_lp = old_logprobs.float().to(new_lp.device)
    mask = action_mask.float().to(new_lp.device)
    ratio = cispo_ratio(new_lp, old_lp)
    weight = ratio.clamp(min=clip_low, max=clip_high).detach()
    adv = advantage if isinstance(advantage, torch.Tensor) else torch.tensor(float(advantage))
    adv = adv.float().to(new_lp.device)
    per_tok = -(weight * adv * new_lp * mask)
    denom = float(normalizer) if normalizer is not None else float(mask.sum().clamp_min(1.0))
    return per_tok.sum() / denom


def joint_loss(
    cispo_loss: torch.Tensor,
    opd_loss: torch.Tensor,
    *,
    lambda_opd: float,
) -> torch.Tensor:
    """L = L_CISPO + lambda * L_OPD (single optimizer step per batch)."""
    return cispo_loss + float(lambda_opd) * opd_loss
