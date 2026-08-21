from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from scape.training.sr_opd_loss import compute_sr_opd_ce, sr_opd_ce_from_logits


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
