"""Synthetic sanity tests for unified classification metrics."""

from __future__ import annotations

import pytest

from inference.scope.eval_common import classification_metrics

KEEP = "KEEP_EVIDENCE"
SKIP = "SKIP_DUPLICATE"


def test_always_predict_keep():
    gold = [KEEP, KEEP, SKIP, SKIP]
    pred = [KEEP, KEEP, KEEP, KEEP]
    m = classification_metrics(gold, pred)
    assert m["recall_KEEP"] == pytest.approx(1.0)
    assert m["recall_SKIP"] == pytest.approx(0.0)
    assert m["f1_KEEP"] < 1.0  # not all gold KEEP


def test_always_predict_skip():
    gold = [KEEP, KEEP, SKIP, SKIP]
    pred = [SKIP, SKIP, SKIP, SKIP]
    m = classification_metrics(gold, pred)
    assert m["recall_SKIP"] == pytest.approx(1.0)
    assert m["recall_KEEP"] == pytest.approx(0.0)
    assert m["f1_SKIP"] < 1.0


def test_perfect_classification():
    gold = [KEEP, KEEP, SKIP, SKIP]
    pred = [KEEP, KEEP, SKIP, SKIP]
    m = classification_metrics(gold, pred)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["f1_KEEP"] == pytest.approx(1.0)
    assert m["f1_SKIP"] == pytest.approx(1.0)


def test_half_random_confusion_matches_manual():
    gold = [KEEP, KEEP, SKIP, SKIP]
    pred = [KEEP, SKIP, KEEP, SKIP]
    m = classification_metrics(gold, pred)
    cm = m["confusion_matrix"]
    assert cm["tp_keep"] == 1
    assert cm["fp_keep"] == 1
    assert cm["fn_keep"] == 1
    assert cm["tp_skip"] == 1
    assert cm["fp_skip"] == 1
    assert cm["fn_skip"] == 1


def test_empty_raises():
    with pytest.raises(ValueError):
        classification_metrics([], [])
