"""Canonical learnability metrics for tool-token OPD (audit 20260813).

M1 = KL(T || S)  — forward KL, teacher distribution, >= 0
M2 = KL(S || T)  — reverse KL, >= 0
M3 = JS(T, S)    — Jensen-Shannon, >= 0
M4 = signed_gap  — mean(log p_T(token) - log p_S(token)) on teacher-forced tokens;
                   may be negative; NOT a divergence.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

NUMERIC_FLOOR = -1e-7


def kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    forward: bool = True,
) -> torch.Tensor:
    """Per-position KL divergence from logits [..., vocab].

    forward=True  -> KL(T || S)
    forward=False -> KL(S || T)
    """
    t_logp = F.log_softmax(teacher_logits, dim=-1)
    s_logp = F.log_softmax(student_logits, dim=-1)
    if forward:
        t_p = t_logp.exp()
        return (t_p * (t_logp - s_logp)).sum(dim=-1)
    s_p = s_logp.exp()
    return (s_p * (s_logp - t_logp)).sum(dim=-1)


def js_from_logits(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Per-position JS(T, S) from logits."""
    t_logp = F.log_softmax(teacher_logits, dim=-1)
    s_logp = F.log_softmax(student_logits, dim=-1)
    t_p = t_logp.exp()
    s_p = s_logp.exp()
    m_p = 0.5 * (t_p + s_p)
    m_logp = m_p.log()
    kl_tm = (t_p * (t_logp - m_logp)).sum(dim=-1)
    kl_sm = (s_p * (s_logp - m_logp)).sum(dim=-1)
    return 0.5 * kl_tm + 0.5 * kl_sm


def signed_logprob_gap(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Per-position signed gap on teacher-forced tokens."""
    if isinstance(token_ids, torch.Tensor):
        ids = token_ids
    else:
        ids = torch.tensor(token_ids, device=teacher_logits.device, dtype=torch.long)
    t_lp = F.log_softmax(teacher_logits, dim=-1)
    s_lp = F.log_softmax(student_logits, dim=-1)
    ids_col = ids.unsqueeze(1)
    t_sel = t_lp.gather(1, ids_col).squeeze(1)
    s_sel = s_lp.gather(1, ids_col).squeeze(1)
    return t_sel - s_sel


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean over masked positions; falls back to unmasked mean if mask empty."""
    if values.numel() == 0:
        return 0.0
    m = mask.to(dtype=values.dtype, device=values.device)
    if float(m.sum().item()) <= 0:
        return float(values.mean().item())
    return float((values * m).sum().item() / m.sum().item())


def aggregate_token_metrics(
    forward_kl: torch.Tensor,
    reverse_kl: torch.Tensor,
    js: torch.Tensor,
    signed_gap: torch.Tensor,
    mask: torch.Tensor,
    *,
    name_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    value_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Aggregate per-token metric tensors with optional span masks."""
    out = {
        "forward_KL": masked_mean(forward_kl, mask),
        "reverse_KL": masked_mean(reverse_kl, mask),
        "JS": masked_mean(js, mask),
        "signed_gap": masked_mean(signed_gap, mask),
        "tool_name_KL": masked_mean(
            forward_kl, name_mask if name_mask is not None else mask
        ),
        "arg_key_KL": masked_mean(
            forward_kl, key_mask if key_mask is not None else mask
        ),
        "arg_value_KL": masked_mean(
            forward_kl, value_mask if value_mask is not None else mask
        ),
    }
    return out


def kl_discrete(p: Mapping[str, float], q: Mapping[str, float], *, eps: float = 1e-12) -> float:
    keys = sorted(set(p) | set(q))
    s = 0.0
    for k in keys:
        pk = max(eps, min(1.0 - eps, float(p.get(k, 0.0))))
        qk = max(eps, min(1.0 - eps, float(q.get(k, 0.0))))
        s += pk * math.log(pk / qk)
    return float(s)


def js_discrete(p: Mapping[str, float], q: Mapping[str, float], *, eps: float = 1e-12) -> float:
    keys = sorted(set(p) | set(q))
    m = {k: 0.5 * (float(p.get(k, 0.0)) + float(q.get(k, 0.0))) for k in keys}
    return 0.5 * kl_discrete(p, m, eps=eps) + 0.5 * kl_discrete(q, m, eps=eps)


def probs_from_logits_list(logits: Sequence[float], labels: Sequence[str]) -> dict[str, float]:
    """Softmax over a small discrete support (for unit-test reference)."""
    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    z = sum(exps)
    return {labels[i]: exps[i] / z for i in range(len(labels))}


def assert_nonnegative(metric: float, name: str) -> None:
    if metric < NUMERIC_FLOOR:
        raise ValueError(f"{name}={metric} < {NUMERIC_FLOOR}")
