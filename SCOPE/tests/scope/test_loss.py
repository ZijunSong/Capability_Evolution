"""Endorse / correct loss unit tests."""

from __future__ import annotations

from training.opd_v2.correct import compute_correct_loss, mean_token_logprob
from training.opd_v2.endorse import compute_endorse_loss


def test_endorse_positive_gap_larger_gate():
    # teacher >> student → larger gate
    out_hi = compute_endorse_loss([-2.0, -2.0], [-0.1, -0.1], beta=5.0)
    out_lo = compute_endorse_loss([-0.1, -0.1], [-2.0, -2.0], beta=5.0)
    assert out_hi.gate > out_lo.gate


def test_endorse_invalid_zero():
    out = compute_endorse_loss([-1.0], [-0.5], validity_mask=0)
    assert out.loss == 0.0


def test_endorse_gate_detached_scalar():
    out = compute_endorse_loss([-1.0, -1.0], [0.0, 0.0], beta=5.0)
    assert isinstance(out.gate, float)
    assert out.loss == -out.gate * (-1.0)


def test_correct_improves_when_recommended_better():
    bad = compute_correct_loss([-2.0, -2.0], [-2.5, -2.5])
    good = compute_correct_loss([-2.0, -2.0], [-0.5, -0.5])
    assert good.loss < bad.loss


def test_length_normalization():
    short = mean_token_logprob([-1.0])
    long = mean_token_logprob([-1.0, -1.0, -1.0])
    assert short == long


def test_correct_invalid_zero():
    out = compute_correct_loss([-1.0], [-0.5], validity_mask=0)
    assert out.loss == 0.0
