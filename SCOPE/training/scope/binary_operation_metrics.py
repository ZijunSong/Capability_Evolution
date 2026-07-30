"""Standard binary operation metrics for KEEP_EVIDENCE / SKIP_DUPLICATE.

All metrics use explicit one-vs-rest confusion counts per class.
Do NOT label recall, class accuracy, or single-class hit rate as F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from harness.capability.dup_operation import DupOperation

KEEP = DupOperation.KEEP_EVIDENCE.value
SKIP = DupOperation.SKIP_DUPLICATE.value


@dataclass(frozen=True)
class ClassMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def to_dict(self, prefix: str) -> dict[str, Any]:
        return {
            f"TP_{prefix}": self.tp,
            f"FP_{prefix}": self.fp,
            f"FN_{prefix}": self.fn,
            f"precision_{prefix}": self.precision,
            f"recall_{prefix}": self.recall,
            f"f1_{prefix}": self.f1,
        }


@dataclass(frozen=True)
class BinaryOperationMetrics:
    tp_keep: int
    fp_keep: int
    fn_keep: int
    tp_skip: int
    fp_skip: int
    fn_skip: int
    precision_keep: float
    recall_keep: float
    f1_keep: float
    precision_skip: float
    recall_skip: float
    f1_skip: float
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    n_samples: int

    @property
    def keep(self) -> ClassMetrics:
        return ClassMetrics(
            self.tp_keep, self.fp_keep, self.fn_keep,
            self.precision_keep, self.recall_keep, self.f1_keep,
        )

    @property
    def skip(self) -> ClassMetrics:
        return ClassMetrics(
            self.tp_skip, self.fp_skip, self.fn_skip,
            self.precision_skip, self.recall_skip, self.f1_skip,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            **self.keep.to_dict("KEEP"),
            **self.skip.to_dict("SKIP"),
            "KEEP_EVIDENCE": {
                "tp": self.tp_keep,
                "fp": self.fp_keep,
                "fn": self.fn_keep,
                "precision": self.precision_keep,
                "recall": self.recall_keep,
                "f1": self.f1_keep,
            },
            "SKIP_DUPLICATE": {
                "tp": self.tp_skip,
                "fp": self.fp_skip,
                "fn": self.fn_skip,
                "precision": self.precision_skip,
                "recall": self.recall_skip,
                "f1": self.f1_skip,
            },
        }
        return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


def compute_binary_operation_metrics(
    gold: Sequence[str],
    pred: Sequence[str],
) -> BinaryOperationMetrics:
    """Compute standard binary metrics from parallel gold/pred label lists."""
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")

    tp_keep = fp_keep = fn_keep = 0
    tp_skip = fp_skip = fn_skip = 0
    correct = 0

    for g, p in zip(gold, pred):
        g = str(g).upper()
        p = str(p).upper()
        if g == p:
            correct += 1

        if g == KEEP:
            if p == KEEP:
                tp_keep += 1
            else:
                fn_keep += 1
            if p == SKIP:
                fp_skip += 1
        elif g == SKIP:
            if p == SKIP:
                tp_skip += 1
            else:
                fn_skip += 1
            if p == KEEP:
                fp_keep += 1

    p_keep, r_keep, f1_keep = _prf(tp_keep, fp_keep, fn_keep)
    p_skip, r_skip, f1_skip = _prf(tp_skip, fp_skip, fn_skip)
    n = len(gold)

    return BinaryOperationMetrics(
        tp_keep=tp_keep,
        fp_keep=fp_keep,
        fn_keep=fn_keep,
        tp_skip=tp_skip,
        fp_skip=fp_skip,
        fn_skip=fn_skip,
        precision_keep=p_keep,
        recall_keep=r_keep,
        f1_keep=f1_keep,
        precision_skip=p_skip,
        recall_skip=r_skip,
        f1_skip=f1_skip,
        macro_f1=(f1_keep + f1_skip) / 2.0,
        balanced_accuracy=(r_keep + r_skip) / 2.0,
        accuracy=correct / max(n, 1),
        n_samples=n,
    )


def accumulate_from_pairs(
    pairs: Iterable[tuple[str | DupOperation | None, str | DupOperation | None]],
) -> BinaryOperationMetrics:
    """Accumulate metrics from (gold, pred) pairs; skip None gold."""
    gold_list: list[str] = []
    pred_list: list[str] = []
    for g, p in pairs:
        if g is None:
            continue
        gold_list.append(str(g).upper() if not isinstance(g, DupOperation) else g.value)
        if p is None:
            pred_list.append("__PARSE_FAIL__")
        else:
            pred_list.append(str(p).upper() if not isinstance(p, DupOperation) else p.value)
    return compute_binary_operation_metrics(gold_list, pred_list)
