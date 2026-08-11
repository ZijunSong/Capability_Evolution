"""Unified tool-call OPD (on-policy distillation) losses and divergence metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _safe_prob(p: float, eps: float = 1e-12) -> float:
    return min(1.0 - eps, max(eps, float(p)))


def normalize_probs(logits_or_probs: Mapping[str, float], *, as_logits: bool = False) -> dict[str, float]:
    names = list(logits_or_probs.keys())
    if not names:
        return {}
    if as_logits:
        m = max(float(logits_or_probs[n]) for n in names)
        exps = {n: math.exp(float(logits_or_probs[n]) - m) for n in names}
        z = sum(exps.values())
        return {n: exps[n] / z for n in names}
    total = sum(max(0.0, float(v)) for v in logits_or_probs.values())
    if total <= 0:
        u = 1.0 / len(names)
        return {n: u for n in names}
    return {n: max(0.0, float(logits_or_probs[n])) / total for n in names}


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float], *, eps: float = 1e-12) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    s = 0.0
    for k in keys:
        pk = _safe_prob(p.get(k, 0.0), eps)
        qk = _safe_prob(q.get(k, 0.0), eps)
        s += pk * math.log(pk / qk)
    return float(s)


def js_divergence(p: Mapping[str, float], q: Mapping[str, float], *, eps: float = 1e-12) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    m = {k: 0.5 * (float(p.get(k, 0.0)) + float(q.get(k, 0.0))) for k in keys}
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def entropy(p: Mapping[str, float], *, eps: float = 1e-12) -> float:
    s = 0.0
    for v in p.values():
        pk = _safe_prob(v, eps)
        s -= pk * math.log(pk)
    return float(s)


def tool_name_divergence(
    student_probs: Mapping[str, float],
    teacher_probs: Mapping[str, float],
    *,
    as_logits: bool = False,
) -> dict[str, float]:
    p = normalize_probs(student_probs, as_logits=as_logits)
    q = normalize_probs(teacher_probs, as_logits=as_logits)
    # Align support
    keys = sorted(set(p) | set(q))
    p = {k: p.get(k, 0.0) for k in keys}
    q = {k: q.get(k, 0.0) for k in keys}
    p = normalize_probs(p)
    q = normalize_probs(q)
    return {
        "I_name_js": js_divergence(p, q),
        "kl_student_teacher": kl_divergence(p, q),
        "kl_teacher_student": kl_divergence(q, p),
        "student_entropy": entropy(p),
        "teacher_entropy": entropy(q),
    }


def token_kl(
    student_logprobs: Sequence[float],
    teacher_logprobs: Sequence[float],
) -> float:
    """Mean token-level KL proxy from aligned log-prob sequences.

    Expects log-probabilities of the *same* teacher-decoded tokens under both
    models (teacher-forced). KL(teacher || student) ≈ mean(log p_t - log p_s).
    """
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError("student/teacher logprob length mismatch")
    if not student_logprobs:
        return 0.0
    diffs = [float(t) - float(s) for s, t in zip(student_logprobs, teacher_logprobs)]
    return float(sum(diffs) / len(diffs))


def tool_opd_loss(
    *,
    tool_token_kl: float,
    anchor_kl: float = 0.0,
    tool_weight: float = 1.0,
    anchor_weight: float = 0.1,
    teacher_confidence: float | None = None,
) -> dict[str, float]:
    w = 1.0
    if teacher_confidence is not None:
        w = max(0.0, min(1.0, float(teacher_confidence)))
    loss = w * (tool_weight * float(tool_token_kl) + anchor_weight * float(anchor_kl))
    return {
        "loss": loss,
        "tool_token_kl": float(tool_token_kl),
        "anchor_kl": float(anchor_kl),
        "weight": w,
    }


def learnability_score(d_pre: float, d_post: float, *, eps: float = 1e-8) -> float:
    """L_m = 1 - D_post / (D_pre + eps)."""
    return float(1.0 - float(d_post) / (float(d_pre) + eps))


def argument_edit_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    keys = set(a) | set(b)
    dist = 0
    for k in keys:
        if a.get(k) != b.get(k):
            dist += 1
    return dist


def disagreement_stats(
    student_call: Mapping[str, Any],
    teacher_call: Mapping[str, Any],
) -> dict[str, Any]:
    s_name = student_call.get("name")
    t_name = teacher_call.get("name")
    s_args = dict(student_call.get("arguments") or {})
    t_args = dict(teacher_call.get("arguments") or {})
    return {
        "tool_name_disagreement": int(s_name != t_name),
        "exact_tool_call_disagreement": int(s_name != t_name or s_args != t_args),
        "argument_edit_distance": argument_edit_distance(s_args, t_args),
    }
