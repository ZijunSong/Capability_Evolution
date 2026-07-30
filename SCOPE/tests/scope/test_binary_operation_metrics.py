"""Unit tests for binary operation metrics (Round 4 Barrier 1.1)."""

from __future__ import annotations

import pytest

from harness.capability.dup_operation import DupOperation
from training.scope.binary_operation_metrics import compute_binary_operation_metrics

KEEP = DupOperation.KEEP_EVIDENCE.value
SKIP = DupOperation.SKIP_DUPLICATE.value


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


class TestCaseA:
    """gold=[KEEP,KEEP,SKIP,SKIP], pred=[KEEP,KEEP,KEEP,KEEP]."""

    @pytest.fixture
    def m(self):
        return compute_binary_operation_metrics(
            [KEEP, KEEP, SKIP, SKIP],
            [KEEP, KEEP, KEEP, KEEP],
        )

    def test_confusion_counts(self, m):
        assert m.tp_keep == 2
        assert m.fp_keep == 2
        assert m.fn_keep == 0
        assert m.tp_skip == 0
        assert m.fp_skip == 0
        assert m.fn_skip == 2

    def test_keep_metrics(self, m):
        p, r, f1 = _prf(2, 2, 0)
        assert m.precision_keep == pytest.approx(p)
        assert m.recall_keep == pytest.approx(r)
        assert m.f1_keep == pytest.approx(f1)

    def test_skip_metrics(self, m):
        p, r, f1 = _prf(0, 0, 2)
        assert m.precision_skip == pytest.approx(p)
        assert m.recall_skip == pytest.approx(r)
        assert m.f1_skip == pytest.approx(f1)

    def test_aggregate(self, m):
        assert m.accuracy == pytest.approx(0.5)
        assert m.balanced_accuracy == pytest.approx(0.5)
        assert m.macro_f1 == pytest.approx((m.f1_keep + m.f1_skip) / 2)
        assert m.recall_skip == 0.0
        assert m.f1_skip == 0.0
        assert m.f1_keep < 1.0
        assert m.macro_f1 < 1.0


class TestCaseB:
    """pred = all SKIP."""

    @pytest.fixture
    def m(self):
        return compute_binary_operation_metrics(
            [KEEP, KEEP, SKIP, SKIP],
            [SKIP, SKIP, SKIP, SKIP],
        )

    def test_confusion_counts(self, m):
        assert m.tp_keep == 0
        assert m.fp_keep == 0
        assert m.fn_keep == 2
        assert m.tp_skip == 2
        assert m.fp_skip == 2
        assert m.fn_skip == 0

    def test_metrics(self, m):
        p_k, r_k, f1_k = _prf(0, 0, 2)
        p_s, r_s, f1_s = _prf(2, 2, 0)
        assert m.precision_keep == pytest.approx(p_k)
        assert m.recall_keep == pytest.approx(r_k)
        assert m.f1_keep == pytest.approx(f1_k)
        assert m.precision_skip == pytest.approx(p_s)
        assert m.recall_skip == pytest.approx(r_s)
        assert m.f1_skip == pytest.approx(f1_s)
        assert m.accuracy == pytest.approx(0.5)
        assert m.balanced_accuracy == pytest.approx(0.5)


class TestCaseC:
    """pred = gold (perfect)."""

    @pytest.fixture
    def m(self):
        gold = [KEEP, KEEP, SKIP, SKIP]
        return compute_binary_operation_metrics(gold, gold)

    def test_perfect(self, m):
        assert m.tp_keep == 2
        assert m.tp_skip == 2
        assert m.fp_keep == 0
        assert m.fn_keep == 0
        assert m.fp_skip == 0
        assert m.fn_skip == 0
        assert m.precision_keep == pytest.approx(1.0)
        assert m.recall_keep == pytest.approx(1.0)
        assert m.f1_keep == pytest.approx(1.0)
        assert m.precision_skip == pytest.approx(1.0)
        assert m.recall_skip == pytest.approx(1.0)
        assert m.f1_skip == pytest.approx(1.0)
        assert m.macro_f1 == pytest.approx(1.0)
        assert m.balanced_accuracy == pytest.approx(1.0)
        assert m.accuracy == pytest.approx(1.0)


def test_all_keep_on_bilateral_data():
    """All-KEEP predictions must yield SKIP recall=0, SKIP F1=0, KEEP F1<1, macro F1<1."""
    m = compute_binary_operation_metrics(
        [KEEP, SKIP, KEEP, SKIP],
        [KEEP, KEEP, KEEP, KEEP],
    )
    assert m.recall_skip == 0.0
    assert m.f1_skip == 0.0
    assert m.f1_keep < 1.0
    assert m.macro_f1 < 1.0
