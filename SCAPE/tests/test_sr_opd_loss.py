from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from scape.training.sr_opd_loss import (
    compute_sr_opd_ce,
    compute_sr_opd_reverse_kl,
    compute_sr_opd_sampled_gap,
    reverse_kl_per_token,
    sr_opd_ce_from_logits,
)


def test_formal_loss_has_no_component_branch():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scape/training/sr_opd_loss.py").read_text(encoding="utf-8")
    hf = (root / "scape/training/hf_tool_opd.py").read_text(encoding="utf-8")
    start = hf.find("def train_projected_step")
    end = hf.find("def train_step", start)
    projected = hf[start:end]
    assert "component_id" not in src
    assert "if component" not in projected
    assert "verify_tool" not in projected
    assert "importance_tagging" not in projected


def test_token_mask_controls_backward():
    torch.manual_seed(0)
    vocab = 8
    steps = 4
    logits = torch.randn(steps, vocab, requires_grad=True)
    targets = torch.tensor([1, 2, 3, 4])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = sr_opd_ce_from_logits(logits, targets, mask)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[1].abs().sum()) == 0.0
    assert float(logits.grad[3].abs().sum()) == 0.0
    assert float(logits.grad[0].abs().sum()) > 0.0
    assert float(logits.grad[2].abs().sum()) > 0.0


def test_weight_controls_backward():
    torch.manual_seed(1)
    logits = torch.randn(3, 6, requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    mask = torch.ones(3)
    weight = torch.tensor([0.0, 2.0, 0.0])
    loss = sr_opd_ce_from_logits(logits, targets, mask, weight)
    loss.backward()
    assert float(logits.grad[0].abs().sum()) == 0.0
    assert float(logits.grad[2].abs().sum()) == 0.0
    assert float(logits.grad[1].abs().sum()) > 0.0


def test_reported_loss_is_same_tensor_as_backward():
    logprobs = torch.tensor([-0.2, -1.4, -0.5], requires_grad=True)
    mask = torch.tensor([1.0, 1.0, 0.0])
    loss = compute_sr_opd_ce(logprobs, mask)
    reported = float(loss.detach().item())
    loss.backward()
    assert abs(reported - float((-(logprobs.detach() * mask)[:2].sum() / 2))) < 1e-5
    assert logprobs.grad is not None


def test_deterministic_overfit_lowers_sr_opd_ce():
    torch.manual_seed(0)
    vocab = 16
    logits = nn.Parameter(torch.zeros(5, vocab))
    opt = torch.optim.SGD([logits], lr=2.0)
    targets = torch.tensor([3, 4, 5, 6, 7])
    mask = torch.ones(5)
    losses: list[float] = []
    exact: list[float] = []
    with torch.no_grad():
        losses.append(float(sr_opd_ce_from_logits(logits, targets, mask)))
        exact.append(float((logits.argmax(dim=-1) == targets).float().mean()))
    for _ in range(40):
        opt.zero_grad()
        loss = sr_opd_ce_from_logits(logits, targets, mask)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        pred = logits.argmax(dim=-1)
        exact.append(float((pred == targets).float().mean()))
    assert losses[-1] < losses[0] * 0.5
    assert exact[-1] > exact[0]
    assert exact[-1] >= 0.8


def test_reverse_kl_matches_manual_and_grads_student_only():
    torch.manual_seed(0)
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8, requires_grad=True)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss = compute_sr_opd_reverse_kl(student, teacher, mask)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert float(student.grad[2].abs().sum()) == 0.0
    assert float(student.grad[0].abs().sum()) > 0.0
    s_logp = torch.nn.functional.log_softmax(student.detach(), dim=-1)
    t_logp = torch.nn.functional.log_softmax(teacher.detach(), dim=-1)
    expected = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    manual = (expected * mask).sum() / mask.sum()
    assert abs(float(loss.detach()) - float(manual)) < 1e-5


def test_reverse_kl_zero_when_distributions_match():
    logits = torch.randn(3, 5)
    mask = torch.ones(3)
    loss = compute_sr_opd_reverse_kl(logits, logits.clone(), mask)
    assert float(loss) < 1e-5
    assert reverse_kl_per_token(logits, logits).abs().max() < 1e-5


def test_sampled_gap_zero_when_logprobs_match():
    lp = torch.tensor([-0.2, -1.1, -0.4], requires_grad=True)
    loss = compute_sr_opd_sampled_gap(lp, lp.detach())
    assert abs(float(loss.detach())) < 1e-6


def test_sampled_gap_grads_student_only_and_token_mean():
    torch.manual_seed(0)
    student = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    teacher = torch.tensor([-0.2, -0.3, -0.4])
    mask = torch.tensor([1.0, 0.0, 1.0])
    loss = compute_sr_opd_sampled_gap(student, teacher, mask, gate_beta=5.0)
    loss.backward()
    assert student.grad is not None
    assert float(student.grad[1].abs()) == 0.0
    assert float(student.grad[0].abs()) > 0.0
    assert float(student.grad[2].abs()) > 0.0
    delta = (teacher - student.detach())
    gate = torch.sigmoid(5.0 * delta)
    gap = teacher - student.detach()
    expected = (gate * gap * mask).sum() / mask.sum()
    assert abs(float(loss.detach()) - float(expected)) < 1e-5
