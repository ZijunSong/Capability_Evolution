"""Action-level SDI losses for SCOPE v3.

L_SDI = -w * log π(target_action | student_state)
Optional KL stabilization against reference policy.

Does NOT depend on teacher logits.

Loss modes:
  sample_normalized_action_ce — per-sample mean CE, then batch mean (Round 2 default)
  legacy_token_ce — global token-mean CE (ablation)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import torch
import torch.nn.functional as F


class LossMode(str, Enum):
    SAMPLE_NORMALIZED_ACTION_CE = "sample_normalized_action_ce"
    LEGACY_TOKEN_CE = "legacy_token_ce"
    OPERATION_CE = "operation_ce"


@dataclass
class SDILossConfig:
    kl_coef: float = 0.01
    ignore_index: int = -100
    label_smoothing: float = 0.0
    loss_mode: LossMode = LossMode.SAMPLE_NORMALIZED_ACTION_CE
    route_balancing: bool = False


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


def _per_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (per_token_ce [B,T], active_mask [B,T], per_sample_mean [B])."""
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
    return per_tok, active, per_sample


def legacy_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Global token-mean CE — long actions dominate gradient mass."""
    per_tok, active, _ = _per_token_ce(
        logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing
    )
    n_active = int(active.sum().item())
    if n_active == 0:
        return per_tok.sum() * 0.0, 0
    return (per_tok * active).sum() / active.sum().clamp(min=1.0), n_active


def route_balance_weights(
    routes: Sequence[str],
    *,
    enabled: bool,
) -> torch.Tensor:
    """Balance ENDORSE / CORRECT sample weights (training only)."""
    if not enabled:
        return torch.ones(len(routes), dtype=torch.float32)
    counts: dict[str, int] = {}
    for r in routes:
        key = str(r).upper()
        if key in {"ENDORSE", "CORRECT"}:
            counts[key] = counts.get(key, 0) + 1
    weights: list[float] = []
    for r in routes:
        key = str(r).upper()
        if key in counts and counts[key] > 0:
            weights.append(1.0 / counts[key])
        else:
            weights.append(1.0)
    w = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight = 1
    return w * (len(w) / w.sum().clamp(min=1e-8))


def operation_balance_weights(
    operations: Sequence[str],
    *,
    enabled: bool,
) -> torch.Tensor:
    """Balance KEEP_EVIDENCE / SKIP_DUPLICATE sample weights."""
    if not enabled:
        return torch.ones(len(operations), dtype=torch.float32)
    counts: dict[str, int] = {}
    for op in operations:
        key = str(op).upper()
        if key in {"KEEP_EVIDENCE", "SKIP_DUPLICATE"}:
            counts[key] = counts.get(key, 0) + 1
    weights: list[float] = []
    for op in operations:
        key = str(op).upper()
        if key in counts and counts[key] > 0:
            weights.append(1.0 / counts[key])
        else:
            weights.append(1.0)
    w = torch.tensor(weights, dtype=torch.float32)
    return w * (len(w) / w.sum().clamp(min=1e-8))


def sdi_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_mode: LossMode = LossMode.SAMPLE_NORMALIZED_ACTION_CE,
) -> tuple[torch.Tensor, int]:
    """Token CE over action span; optional per-sample weights.

    logits: [B, T, V], labels: [B, T]
    """
    if loss_mode == LossMode.LEGACY_TOKEN_CE:
        return legacy_token_cross_entropy(
            logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing
        )

    _per_tok, active, per_sample = _per_token_ce(
        logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing
    )

    if sample_weights is None:
        weights = torch.ones_like(per_sample)
    else:
        weights = sample_weights.to(per_sample.dtype).to(per_sample.device)

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
    routes: Sequence[str] | None = None,
    ref_logits: torch.Tensor | None = None,
    config: SDILossConfig | None = None,
) -> SDILossOutput:
    cfg = config or SDILossConfig()
    weights = sample_weights
    if routes and cfg.route_balancing:
        rb = route_balance_weights(routes, enabled=True).to(logits.device)
        if weights is None:
            weights = rb
        else:
            weights = weights.to(rb.device) * rb
    sdi, n_active = sdi_cross_entropy(
        logits,
        labels,
        sample_weights=weights,
        ignore_index=cfg.ignore_index,
        label_smoothing=cfg.label_smoothing,
        loss_mode=cfg.loss_mode,
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
