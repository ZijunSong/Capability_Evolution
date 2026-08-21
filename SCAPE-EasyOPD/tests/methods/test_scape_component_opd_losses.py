from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from easyopd.methods.scape_component_opd.losses.action_ce import masked_action_ce
from easyopd.methods.scape_component_opd.losses.forward_kl import forward_kl_exact
from easyopd.methods.scape_component_opd.losses.jsd import alpha_jsd
from easyopd.methods.scape_component_opd.losses.projected_action_ce import projected_action_ce
from easyopd.methods.scape_component_opd.losses.reverse_kl import reverse_kl_exact


def _manual_forward(student_logits, teacher_logits):
    sl = F.log_softmax(student_logits.float(), dim=-1)
    tl = F.log_softmax(teacher_logits.float(), dim=-1)
    return (tl.exp() * (tl - sl)).sum(dim=-1)


def _manual_reverse(student_logits, teacher_logits):
    sl = F.log_softmax(student_logits.float(), dim=-1)
    tl = F.log_softmax(teacher_logits.float(), dim=-1)
    return (sl.exp() * (sl - tl)).sum(dim=-1)


def test_forward_kl_matches_manual_probability_reference():
    s = torch.tensor([[[0.1, 0.2, -0.3], [1.0, -1.0, 0.0]]])
    t = torch.tensor([[[0.2, -0.1, 0.5], [0.0, 0.5, -0.5]]])
    mask = torch.tensor([[1.0, 0.0]])
    loss, metrics = forward_kl_exact(s, t, mask)
    expected = _manual_forward(s, t)[0, 0]
    assert torch.allclose(loss, expected, atol=1e-6)
    assert metrics["n_tokens"] == 1.0


def test_reverse_kl_matches_manual_and_has_gradient():
    s = torch.randn(2, 3, 5, requires_grad=True)
    t = torch.randn(2, 3, 5)
    loss, _ = reverse_kl_exact(s, t)
    expected = _manual_reverse(s, t).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert float(s.grad.abs().sum()) > 0


def test_reverse_kl_finite_difference_tiny_logits():
    s = torch.tensor([[[0.2, -0.1, 0.4]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 0.3, -0.2]]], dtype=torch.float64)
    loss, _ = reverse_kl_exact(s, t)
    loss.backward()
    eps = 1e-4
    idx = (0, 0, 1)
    sp = s.detach().clone(); sm = s.detach().clone()
    sp[idx] += eps; sm[idx] -= eps
    lp, _ = reverse_kl_exact(sp, t)
    lm, _ = reverse_kl_exact(sm, t)
    fd = (lp - lm) / (2 * eps)
    assert torch.allclose(s.grad[idx].float(), fd.float(), atol=1e-3)


def test_jsd_and_bf16_are_finite():
    s = torch.tensor([[[20.0, -20.0, 0.0]]], dtype=torch.bfloat16)
    t = torch.tensor([[[-20.0, 20.0, 0.0]]], dtype=torch.bfloat16)
    loss, metrics = alpha_jsd(s, t, alpha=0.5)
    assert torch.isfinite(loss)
    assert metrics["jsd"] >= 0.0


def test_action_ce_and_projected_action_metadata():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    target = torch.tensor([[0, 1]])
    mask = torch.tensor([[1.0, 1.0]])
    loss, metrics = masked_action_ce(logits, target, mask)
    expected = -F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()
    assert torch.allclose(loss, expected)
    ploss, pmetrics = projected_action_ce(logits, target, mask)
    assert torch.allclose(ploss, loss)
    assert pmetrics["target_source"] == "harness_effect_projection"
    assert pmetrics["on_policy_state"] is True
