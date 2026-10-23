"""test_loss_and_grad_parity: new loss pure functions vs the legacy reference.

The legacy ``sr_opd_loss`` implementation is embedded here (migrated for
numerical parity only) so this package stays standalone — no import of the old
research tree.  Loss and gradients must match within tolerance; Teacher/gate
must carry no gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rsi.opd import losses

# ---------------------------------------------------------------------------
# Legacy reference (migrated verbatim from trim.training.sr_opd_loss with its
# kl_from_logits dependency; behavior kept identical).
# ---------------------------------------------------------------------------


def _legacy_compute_sr_opd_ce(logprobs, token_mask, token_weight=None):
    if logprobs.numel() == 0:
        return logprobs.sum() * 0.0
    mask = token_mask.to(device=logprobs.device, dtype=logprobs.dtype)
    if token_weight is None:
        weight = mask
    else:
        weight = mask * token_weight.to(device=logprobs.device, dtype=logprobs.dtype)
    denom = weight.sum().clamp_min(1e-8)
    return -(logprobs * weight).sum() / denom


def _legacy_action_gap_gate(delta, *, gate_beta=5.0, gate_mode="sigmoid"):
    if gate_mode == "sigmoid":
        return torch.sigmoid(delta.detach() * float(gate_beta))
    if gate_mode == "none":
        return torch.ones_like(delta)
    raise ValueError


def _legacy_gated_action_gap_per_token(student_logprobs, teacher_logprobs, *, gate_beta=5.0, gate_mode="sigmoid"):
    teacher = teacher_logprobs.detach().to(
        device=student_logprobs.device, dtype=student_logprobs.dtype
    ).reshape(-1)
    student = student_logprobs.reshape(-1)
    if teacher.numel() != student.numel():
        raise ValueError("length mismatch")
    delta = (teacher - student).detach()
    gate = _legacy_action_gap_gate(delta, gate_beta=gate_beta, gate_mode=gate_mode)
    return gate * (teacher - student)


def _legacy_cispo_token_loss(new_logprobs, old_logprobs, advantage, action_mask, *, clip_low=0.8, clip_high=1.2, normalizer=None):
    new_lp = new_logprobs.float()
    old_lp = old_logprobs.float().to(new_lp.device)
    mask = action_mask.float().to(new_lp.device)
    ratio = (new_lp - old_lp).exp()
    weight = ratio.clamp(min=clip_low, max=clip_high).detach()
    adv = torch.tensor(float(advantage)).float().to(new_lp.device)
    per_tok = -(weight * adv * new_lp * mask)
    denom = float(normalizer) if normalizer is not None else float(mask.sum().clamp_min(1.0))
    return per_tok.sum() / denom


# ---------------------------------------------------------------------------

torch.manual_seed(0)


def test_ce_parity():
    lp = torch.randn(7, requires_grad=True)
    mask = torch.tensor([1, 1, 1, 0, 1, 0, 1], dtype=torch.float32)
    w = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 0.25])
    new = losses.compute_sr_opd_ce(lp, mask, w)
    old = _legacy_compute_sr_opd_ce(lp, mask, w)
    assert torch.allclose(new, old, atol=1e-6)
    (new * 2).backward()
    grad_new = lp.grad.clone()
    lp.grad = None
    (old * 2).backward()
    assert torch.allclose(grad_new, lp.grad, atol=1e-6)


def test_gated_gap_parity_and_no_teacher_grad():
    s = torch.randn(5, requires_grad=True)
    t = torch.randn(5, requires_grad=True)  # teacher requires grad on purpose
    new = losses.gated_action_gap_per_token(s, t)
    old = _legacy_gated_action_gap_per_token(s, t)
    assert torch.allclose(new, old, atol=1e-6)
    new.sum().backward()
    assert t.grad is None or torch.allclose(t.grad, torch.zeros_like(t))
    assert s.grad is not None and s.grad.abs().sum() > 0
    # gate is stop-gradient: gradient of the gap term equals -g * dL/dgap form
    assert torch.allclose(s.grad, -torch.ones_like(s) * torch.sigmoid(((t - s).detach() * 5.0)), atol=1e-5)


def test_gate_beta_zero_is_half_not_none():
    delta = torch.tensor([1.0, -2.0])
    g0 = losses.action_gap_gate(delta, gate_beta=0.0)
    assert torch.allclose(g0, torch.full((2,), 0.5))
    gnone = losses.action_gap_gate(delta, gate_mode="none")
    assert torch.allclose(gnone, torch.ones(2))


def test_projected_gap_formula():
    s = torch.tensor([0.0, -1.0, -2.0], requires_grad=True)
    t = torch.tensor([-0.5, -0.5, -3.0])
    mask = torch.ones(3)
    w = torch.tensor([1.0, 1.0, 2.0])
    loss = losses.gated_projected_gap_loss(s, t, mask, w, gate_beta=5.0)
    gap = torch.sigmoid((t - s).detach() * 5.0) * (t - s)
    expected = (w * gap).sum() / w.sum()
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert s.grad.abs().sum() > 0


def test_zero_denominator_skips_without_dummy_target():
    s = torch.zeros(3, requires_grad=True)
    t = torch.zeros(3)
    mask = torch.zeros(3)
    loss = losses.gated_projected_gap_loss(s, t, mask)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.allclose(s.grad, torch.zeros(3))


def test_cispo_parity():
    new_lp = torch.randn(6, requires_grad=True)
    old_lp = torch.randn(6)
    mask = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.float32)
    loss_new = losses.cispo_token_loss(new_lp, old_lp, 0.7, mask, normalizer=4.0)
    loss_old = _legacy_cispo_token_loss(new_lp, old_lp, 0.7, mask, normalizer=4.0)
    assert torch.allclose(loss_new, loss_old, atol=1e-6)
    # weight is stop-grad clamp(ratio)
    ratio = (new_lp - old_lp).exp()
    w = ratio.clamp(0.8, 1.2).detach()
    expected = -(w * 0.7 * new_lp * mask).sum() / 4.0
    assert torch.allclose(loss_new, expected, atol=1e-6)


def test_length_mismatch_is_error_not_truncation():
    s = torch.randn(4, requires_grad=True)
    t = torch.randn(5)
    try:
        losses.gated_action_gap_per_token(s, t)
    except ValueError:
        pass
    else:
        raise AssertionError("silent truncation must be forbidden")


def test_joint_loss_is_linear_combination():
    a = torch.tensor(1.5)
    b = torch.tensor(2.0)
    assert abs(float(losses.joint_loss(a, b, lambda_opd=0.01)) - (1.5 + 0.01 * 2.0)) < 1e-6


def test_scale_projected_gap_zero_z_gap():
    numer = torch.tensor(3.0, requires_grad=True)
    out = losses.scale_projected_gap_loss(numer, lambda_opd=0.1, z_gap=0.0, world_size=2)
    assert float(out.detach()) == 0.0
    out2 = losses.scale_projected_gap_loss(numer, lambda_opd=0.1, z_gap=4.0, world_size=2)
    assert abs(float(out2.detach()) - 2 * 0.1 * 3.0 / 4.0) < 1e-6
