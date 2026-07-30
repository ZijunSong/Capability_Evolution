"""Round 6 metrics: AUROC, calibration, direct admission behavior."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from harness.capability.dup_operation import DupOperation

KEEP = DupOperation.KEEP_EVIDENCE.value
SKIP = DupOperation.SKIP_DUPLICATE.value


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = int(p * (len(s) - 1))
    return s[idx]


def margin_stats(margins: list[float]) -> dict[str, float]:
    if not margins:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    mean = sum(margins) / len(margins)
    var = sum((x - mean) ** 2 for x in margins) / max(len(margins), 1)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "median": _percentile(margins, 0.5),
        "p01": _percentile(margins, 0.01),
        "p05": _percentile(margins, 0.05),
        "p25": _percentile(margins, 0.25),
        "p50": _percentile(margins, 0.50),
        "p75": _percentile(margins, 0.75),
        "p95": _percentile(margins, 0.95),
        "p99": _percentile(margins, 0.99),
        "n": len(margins),
    }


def auroc(labels: list[int], scores: list[float]) -> float:
    """labels: 1=duplicate(SKIP positive), 0=unique."""
    if not labels:
        return 0.0
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def auprc(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    auc = 0.0
    prev_rec = 0.0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        rec = tp / n_pos
        prec = tp / max(tp + fp, 1)
        auc += (rec - prev_rec) * prec
        prev_rec = rec
    return auc


def predict_at_threshold(margins: list[float], threshold: float) -> list[str]:
    return [SKIP if m >= threshold else KEEP for m in margins]


def direct_behavior_metrics(
    labels: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    """labels/predictions use KEEP_EVIDENCE / SKIP_DUPLICATE."""
    n_dup = n_unique = 0
    dup_reject = dup_keep = unique_keep = unique_skip = 0
    for lab, pred in zip(labels, predictions):
        if lab == SKIP:
            n_dup += 1
            if pred == SKIP:
                dup_reject += 1
            else:
                dup_keep += 1
        elif lab == KEEP:
            n_unique += 1
            if pred == KEEP:
                unique_keep += 1
            else:
                unique_skip += 1
    dup_reject_recall = dup_reject / max(n_dup, 1)
    unique_keep_recall = unique_keep / max(n_unique, 1)
    false_skip_rate = unique_skip / max(n_unique, 1)
    balanced_acc = (dup_reject_recall + unique_keep_recall) / 2
    n_skip_pred = sum(1 for p in predictions if p == SKIP)
    return {
        "n": len(labels),
        "n_duplicate": n_dup,
        "n_unique": n_unique,
        "DupRejectRecall": dup_reject_recall,
        "UniqueKeepRecall": unique_keep_recall,
        "FalseSkipRate": false_skip_rate,
        "DupKeepRate": 1 - dup_reject_recall,
        "BalancedAcc": balanced_acc,
        "n_pred_SKIP": n_skip_pred,
        "predicted_SKIP_prior": n_skip_pred / max(len(predictions), 1),
    }


def best_dup_reject_at_fsr(
    labels: list[str],
    margins: list[float],
    max_fsr: float,
) -> dict[str, float]:
    thresholds = sorted(set(margins))
    if not thresholds:
        thresholds = [0.0]
    best_rec = -1.0
    best_tau = 0.0
    for tau in thresholds:
        preds = predict_at_threshold(margins, tau)
        m = direct_behavior_metrics(labels, preds)
        if m["FalseSkipRate"] <= max_fsr and m["DupRejectRecall"] > best_rec:
            best_rec = m["DupRejectRecall"]
            best_tau = tau
    if best_rec < 0:
        return {"threshold": 0.0, "DupRejectRecall": 0.0, "FalseSkipRate": 1.0}
    preds = predict_at_threshold(margins, best_tau)
    m = direct_behavior_metrics(labels, preds)
    return {
        "threshold": best_tau,
        "DupRejectRecall": m["DupRejectRecall"],
        "FalseSkipRate": m["FalseSkipRate"],
        "BalancedAcc": m["BalancedAcc"],
    }


def brier_score(labels: list[int], probs: list[float]) -> float | None:
    if not labels or not probs:
        return None
    return sum((p - l) ** 2 for p, l in zip(probs, labels)) / len(labels)


def ece(labels: list[int], probs: list[float], n_bins: int = 10) -> float | None:
    if not labels or not probs:
        return None
    bins = [[] for _ in range(n_bins)]
    for l, p in zip(labels, probs):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((l, p))
    ece = 0.0
    n = len(labels)
    for b in bins:
        if not b:
            continue
        acc = sum(l for l, _ in b) / len(b)
        conf = sum(p for _, p in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
    return ece


@dataclass
class ScoredRow:
    label: str
    margin: float
    score_keep: float
    score_skip: float
    prediction: str


def aggregate_scored_rows(rows: list[ScoredRow], threshold: float = 0.0) -> dict[str, Any]:
    labels = [r.label for r in rows]
    margins = [r.margin for r in rows]
    dup_labels = [1 if l == SKIP else 0 for l in labels]
    preds = predict_at_threshold(margins, threshold)
    dup_margins = [m for r, m in zip(rows, margins) if r.label == SKIP]
    uniq_margins = [m for r, m in zip(rows, margins) if r.label == KEEP]
    probs = []
    for m in margins:
        # crude sigmoid mapping for Brier/ECE only when needed
        probs.append(1.0 / (1.0 + math.exp(-m)))
    out: dict[str, Any] = {
        "AUROC": auroc(dup_labels, margins),
        "AUPRC": auprc(dup_labels, margins),
        **direct_behavior_metrics(labels, preds),
        "margin_all": margin_stats(margins),
        "margin_duplicate": margin_stats(dup_margins),
        "margin_unique": margin_stats(uniq_margins),
        "best_DupReject_FSR1": best_dup_reject_at_fsr(labels, margins, 0.01),
        "best_DupReject_FSR5": best_dup_reject_at_fsr(labels, margins, 0.05),
        "best_DupReject_FSR10": best_dup_reject_at_fsr(labels, margins, 0.10),
        "Brier": brier_score(dup_labels, probs),
        "ECE": ece(dup_labels, probs),
        "threshold_zero": threshold,
    }
    return out
