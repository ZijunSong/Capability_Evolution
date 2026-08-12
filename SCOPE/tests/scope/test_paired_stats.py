"""Tests for paired query statistics."""

from __future__ import annotations

import pytest

from inference.scope.paired_stats import paired_query_stats, seed_mean_std


def test_paired_basic():
    base = {"q1": {"recall": 0.0}, "q2": {"recall": 1.0}}
    other = {"q1": {"recall": 1.0}, "q2": {"recall": 1.0}}
    s = paired_query_stats(base, other, metric="recall")
    assert s["n"] == 2
    assert s["mean_delta"] == pytest.approx(0.5)
    assert s["win"] == 1
    assert s["tie"] == 1
    assert s["loss"] == 0


def test_missing_query_raises():
    base = {"q1": 1.0, "q2": 0.0}
    other = {"q1": 1.0}
    with pytest.raises(ValueError):
        paired_query_stats(base, other, require_complete=True)


def test_seed_mean_std():
    s = seed_mean_std([0.2, 0.4, 0.6])
    assert s["mean"] == pytest.approx(0.4)
    assert s["n"] == 3
