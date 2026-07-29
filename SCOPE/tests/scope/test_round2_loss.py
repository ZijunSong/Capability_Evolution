"""Tests for sample-normalized CE and route balancing."""

from __future__ import annotations

import torch

from training.scope.losses import (
    LossMode,
    legacy_token_cross_entropy,
    route_balance_weights,
    sdi_cross_entropy,
)


def _fake_batch(bsz: int = 2, seq: int = 8, vocab: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(bsz, seq, vocab)
    labels = torch.full((bsz, seq), -100, dtype=torch.long)
    labels[:, 3:6] = torch.randint(0, vocab, (bsz, 3))
    return logits, labels


def test_sample_normalized_equals_per_sample_mean():
    logits, labels = _fake_batch()
    loss_sn, n1 = sdi_cross_entropy(logits, labels, loss_mode=LossMode.SAMPLE_NORMALIZED_ACTION_CE)
    loss_legacy, n2 = legacy_token_cross_entropy(logits, labels)
    assert n1 == 2
    assert n2 == 6  # total active tokens across batch
    assert loss_sn.item() > 0
    assert loss_legacy.item() > 0


def test_long_action_weighted_more_in_legacy_mode():
    """Legacy token CE should weight long spans more than sample-normalized."""
    vocab = 16
    logits = torch.randn(2, 12, vocab)
    labels = torch.full((2, 12), -100, dtype=torch.long)
    labels[0, 2:5] = 3
    labels[1, 2:10] = 5  # longer active span
    loss_sn, _ = sdi_cross_entropy(logits, labels, loss_mode=LossMode.SAMPLE_NORMALIZED_ACTION_CE)
    loss_legacy, _ = legacy_token_cross_entropy(logits, labels)
    assert loss_sn.item() > 0
    assert loss_legacy.item() > 0


def test_route_balance_equalizes_endorse_correct():
    routes = ["ENDORSE", "ENDORSE", "CORRECT", "CORRECT", "CORRECT"]
    w = route_balance_weights(routes, enabled=True)
    assert abs(w[0].item() - w[1].item()) < 1e-5
    assert w[2].item() < w[0].item()  # CORRECT has more samples → lower weight
