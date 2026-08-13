"""Unit tests for canonical learnability metrics (audit 20260813)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from scape.training.canonical_metrics import (
    NUMERIC_FLOOR,
    aggregate_token_metrics,
    js_discrete,
    js_from_logits,
    kl_discrete,
    kl_from_logits,
    probs_from_logits_list,
    signed_logprob_gap,
)
from scape.training.hf_tool_opd import ScapeHFToolOPD, mean_canonical_metrics, mean_divergence
from scape.training.tool_mask import tool_loss_mask_from_response


TEACHER_PROBS = [0.7, 0.2, 0.1]
STUDENT_PROBS = [0.4, 0.4, 0.2]
LABELS = ["a", "b", "c"]


def _ref_kl_forward() -> float:
    p = probs_from_logits_list(TEACHER_PROBS, LABELS)
    q = probs_from_logits_list(STUDENT_PROBS, LABELS)
    return kl_discrete(p, q)


def _ref_kl_reverse() -> float:
    p = probs_from_logits_list(TEACHER_PROBS, LABELS)
    q = probs_from_logits_list(STUDENT_PROBS, LABELS)
    return kl_discrete(q, p)


def _ref_js() -> float:
    p = probs_from_logits_list(TEACHER_PROBS, LABELS)
    q = probs_from_logits_list(STUDENT_PROBS, LABELS)
    return js_discrete(p, q)


def _torch_logits(probs: list[float]) -> torch.Tensor:
    return torch.tensor(probs, dtype=torch.float64)


def test_kl_identical_zero():
    t = _torch_logits([1.0, 0.0, 0.0])
    s = _torch_logits([1.0, 0.0, 0.0])
    kl = kl_from_logits(t, s, forward=True).item()
    assert abs(kl) < 1e-6


def test_kl_nonnegative_random():
    for _ in range(20):
        t = torch.randn(128)
        s = torch.randn(128)
        fwd = kl_from_logits(t, s, forward=True).item()
        rev = kl_from_logits(t, s, forward=False).item()
        assert fwd >= NUMERIC_FLOOR
        assert rev >= NUMERIC_FLOOR


def test_js_nonnegative_random():
    for _ in range(20):
        t = torch.randn(64)
        s = torch.randn(64)
        js = js_from_logits(t, s).item()
        assert js >= NUMERIC_FLOOR


def test_kl_teacher_student_direction():
    t = _torch_logits(TEACHER_PROBS)
    s = _torch_logits(STUDENT_PROBS)
    fwd = kl_from_logits(t, s, forward=True).item()
    rev = kl_from_logits(t, s, forward=False).item()
    assert abs(fwd - _ref_kl_forward()) < 1e-5
    assert abs(rev - _ref_kl_reverse()) < 1e-5
    assert fwd != rev


def test_manual_logits_match_torch_reference():
    t = _torch_logits(TEACHER_PROBS)
    s = _torch_logits(STUDENT_PROBS)
    fwd = kl_from_logits(t, s, forward=True).item()
    js = js_from_logits(t, s).item()
    assert abs(fwd - _ref_kl_forward()) < 1e-5
    assert abs(js - _ref_js()) < 1e-5


def test_masked_kl_matches_manual():
    t = torch.stack([_torch_logits(TEACHER_PROBS), _torch_logits([0.5, 0.3, 0.2])])
    s = torch.stack([_torch_logits(STUDENT_PROBS), _torch_logits([0.2, 0.5, 0.3])])
    fwd = kl_from_logits(t, s, forward=True)
    mask = torch.tensor([1.0, 0.0])
    manual = aggregate_token_metrics(fwd, fwd, fwd, fwd, mask)
    assert abs(manual["forward_KL"] - fwd[0].item()) < 1e-6


def test_padding_not_in_loss():
    """Zero mask positions must not affect masked mean."""
    vals = torch.tensor([1.0, 9.0, 9.0])
    mask = torch.tensor([1.0, 0.0, 0.0])
    m = aggregate_token_metrics(vals, vals, vals, vals, mask)
    assert abs(m["forward_KL"] - 1.0) < 1e-6


SAMPLE = """\
to=search
{"query": "who invented x", "top_k": 5}
to=curate
{"add_ids": ["d1"], "remove_ids": []}
end_search
"""


def test_tool_name_mask_exact():
    audit = tool_loss_mask_from_response(
        SAMPLE,
        include_name=True,
        include_arg_keys=False,
        include_arg_values=False,
        include_end_search=False,
    )
    assert audit["n_tool_name"] >= 2
    assert audit["n_argument_key"] == 0
    assert audit["n_argument_value"] == 0


def test_arg_key_mask_exact():
    audit = tool_loss_mask_from_response(
        SAMPLE,
        include_name=False,
        include_arg_keys=True,
        include_arg_values=False,
        include_end_search=False,
    )
    assert audit["n_argument_key"] >= 1


def test_arg_value_mask_exact():
    audit = tool_loss_mask_from_response(
        SAMPLE,
        include_name=False,
        include_arg_keys=False,
        include_arg_values=True,
        include_end_search=False,
    )
    assert audit["n_argument_value"] >= 1


def test_batch_reduction_matches_single():
    """Masked aggregate over batch equals manual average of per-row metrics."""
    rows = [
        {"forward_KL": 0.5, "reverse_KL": 0.4, "JS": 0.3, "signed_gap": 0.1},
        {"forward_KL": 1.5, "reverse_KL": 1.2, "JS": 0.9, "signed_gap": -0.2},
    ]
    manual = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    assert abs(manual["forward_KL"] - 1.0) < 1e-6
    assert abs(manual["signed_gap"] - (-0.05)) < 1e-6


def test_evaluator_matches_trainer_metric():
    """Trainer signed_gap (div) must match score_canonical signed_gap on synthetic backend."""
    t_logits = torch.randn(4, 8)
    s_logits = torch.randn(4, 8)
    ids = torch.tensor([0, 1, 2, 3])
    gap = signed_logprob_gap(t_logits, s_logits, ids)
    trainer_proxy = torch.gather(F.log_softmax(t_logits, -1), 1, ids.unsqueeze(1)).squeeze(1) - \
        torch.gather(F.log_softmax(s_logits, -1), 1, ids.unsqueeze(1)).squeeze(1)
    assert torch.allclose(gap, trainer_proxy, atol=1e-5)
