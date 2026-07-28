"""Action-level SDI losses for SCOPE v3.

L_SDI = -w * log π(target_action | student_state)
Optional KL stabilization against reference policy.

Does NOT depend on teacher logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class SDILossConfig:
    kl_coef: float = 0.01
    ignore_index: int = -100
    label_smoothing: float = 0.0


@dataclass
class SDILossOutput:
    loss: torch.Tensor
    sdi_loss: torch.Tensor
    kl_loss: torch.Tensor
    n_active: int
    metrics: dict[str, float]


def action_span_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_start: Sequence[int] | torch.Tensor,
    action_end: Sequence[int] | torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build labels where only [action_start, action_end) tokens contribute to CE.

    Standard causal LM shift is applied by the caller / model; labels match input_ids
    shape with ignore_index outside the action span.
    """
    labels = input_ids.clone()
    labels[:] = ignore_index
    bsz, _seq = input_ids.shape
    for i in range(bsz):
        s = int(action_start[i])
        e = int(action_end[i])
        if e > s:
            labels[i, s:e] = input_ids[i, s:e]
            # Mask padding inside span
            pad = attention_mask[i, s:e] == 0
            labels[i, s:e][pad] = ignore_index
    return labels


def sdi_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Token CE over action span; optional per-sample weights.

    logits: [B, T, V], labels: [B, T]
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    bsz, seq, vocab = shift_logits.shape
    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.view(-1)
    per_tok = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    ).view(bsz, seq)

    active = (shift_labels != ignore_index).float()
    tok_counts = active.sum(dim=1).clamp(min=1.0)
    per_sample = (per_tok * active).sum(dim=1) / tok_counts

    if sample_weights is None:
        weights = torch.ones_like(per_sample)
    else:
        weights = sample_weights.to(per_sample.dtype).to(per_sample.device)

    # Only samples with at least one active token and positive weight
    mask = (active.sum(dim=1) > 0) & (weights > 0)
    n_active = int(mask.sum().item())
    if n_active == 0:
        return per_sample.sum() * 0.0, 0
    loss = (per_sample * weights * mask.float()).sum() / weights[mask].sum().clamp(min=1e-8)
    return loss, n_active


def kl_to_reference(
    student_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-mean KL(student || ref) on action-span positions."""
    shift_s = student_logits[:, :-1, :].contiguous()
    shift_r = ref_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    active = shift_labels != ignore_index
    if not active.any():
        return student_logits.sum() * 0.0

    log_p = F.log_softmax(shift_s, dim=-1)
    log_q = F.log_softmax(shift_r, dim=-1)
    # KL(P||Q) = sum P * (logP - logQ)
    p = log_p.exp()
    kl_tok = (p * (log_p - log_q)).sum(dim=-1)
    return kl_tok[active].mean()


def compute_sdi_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    ref_logits: torch.Tensor | None = None,
    config: SDILossConfig | None = None,
) -> SDILossOutput:
    cfg = config or SDILossConfig()
    sdi, n_active = sdi_cross_entropy(
        logits,
        labels,
        sample_weights=sample_weights,
        ignore_index=cfg.ignore_index,
        label_smoothing=cfg.label_smoothing,
    )
    if ref_logits is not None and cfg.kl_coef > 0:
        kl = kl_to_reference(
            logits, ref_logits, labels, ignore_index=cfg.ignore_index
        )
    else:
        kl = sdi * 0.0

    loss = sdi + cfg.kl_coef * kl
    return SDILossOutput(
        loss=loss,
        sdi_loss=sdi.detach(),
        kl_loss=kl.detach() if isinstance(kl, torch.Tensor) else torch.tensor(0.0),
        n_active=n_active,
        metrics={
            "sdi_loss": float(sdi.detach().item()) if n_active else 0.0,
            "kl_loss": float(kl.detach().item()) if isinstance(kl, torch.Tensor) else 0.0,
            "n_active": float(n_active),
        },
    )


# Optional pairwise ablation (not Round-1 main path)
def pairwise_ablation_loss(
    logprob_positive: torch.Tensor,
    logprob_negative: torch.Tensor,
) -> torch.Tensor:
    """-log σ(logπ(a+) - logπ(a-)) — ablation only."""
    return -F.logsigmoid(logprob_positive - logprob_negative).mean()
