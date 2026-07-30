"""Round 5 B2 — gradient-sign and objective math unit tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from harness.capability.dup_operation import DupOperation


def _ce_grad_signs(s_keep_init: float, s_skip_init: float, target_idx: int) -> tuple[float, float]:
    s_keep = torch.tensor(s_keep_init, requires_grad=True)
    s_skip = torch.tensor(s_skip_init, requires_grad=True)
    logits = torch.stack([s_keep, s_skip])
    loss = -F.log_softmax(logits, dim=0)[target_idx]
    loss.backward()
    return float(s_keep.grad), float(s_skip.grad)


def _margin_grad_signs(s_keep_init: float, s_skip_init: float, target: DupOperation) -> tuple[float, float]:
    s_keep = torch.tensor(s_keep_init, requires_grad=True)
    s_skip = torch.tensor(s_skip_init, requires_grad=True)
    margin = s_skip - s_keep
    loss = F.softplus(margin) if target == DupOperation.KEEP_EVIDENCE else F.softplus(-margin)
    loss.backward()
    return float(s_keep.grad), float(s_skip.grad)


def test_discriminative_ce_keep_gradient_signs():
    gk, gs = _ce_grad_signs(1.0, 2.0, target_idx=0)
    assert gk < 0, "KEEP: dL/ds_keep < 0"
    assert gs > 0, "KEEP: dL/ds_skip > 0"


def test_discriminative_ce_skip_gradient_signs():
    gk, gs = _ce_grad_signs(1.0, 2.0, target_idx=1)
    assert gk > 0, "SKIP: dL/ds_keep > 0"
    assert gs < 0, "SKIP: dL/ds_skip < 0"


def test_pairwise_margin_keep_gradient_signs():
    gk, gs = _margin_grad_signs(0.5, -0.2, DupOperation.KEEP_EVIDENCE)
    assert gk < 0
    assert gs > 0


def test_pairwise_margin_skip_gradient_signs():
    gk, gs = _margin_grad_signs(0.5, -0.2, DupOperation.SKIP_DUPLICATE)
    assert gk > 0
    assert gs < 0


def test_discriminative_ce_is_softmax_not_negative_target_score():
    """O0/O1 must be CE over [s_keep, s_skip], not -s_target alone."""
    s_keep = torch.tensor(2.0, requires_grad=True)
    s_skip = torch.tensor(1.0, requires_grad=True)
    neg_target = -s_keep
    neg_target.backward()
    assert s_keep.grad.item() == -1.0  # wrong objective would only push s_keep up

    s_keep2 = torch.tensor(2.0, requires_grad=True)
    s_skip2 = torch.tensor(1.0, requires_grad=True)
    ce_keep = -F.log_softmax(torch.stack([s_keep2, s_skip2]), dim=0)[0]
    ce_keep.backward()
    assert s_keep2.grad.item() < 0
    assert s_skip2.grad.item() > 0
