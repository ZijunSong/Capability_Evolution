"""Harness-1 quality for RSI. Final-answer recall is selected-document recall."""

from __future__ import annotations

from typing import Iterable

from trim.eval.harness1_metrics import set_recall


def final_answer_recall(selected_doc_ids: Iterable[str], gold_doc_ids: Iterable[str]) -> float:
    return float(set_recall(selected_doc_ids, gold_doc_ids))
