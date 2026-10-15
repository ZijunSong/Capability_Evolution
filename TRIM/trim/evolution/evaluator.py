"""External scorer. The proposer does not assign its own reward."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from trim.evolution.operators import Proposal
from trim.rsi.metrics import final_answer_recall


class ScoreFn(Protocol):
    def __call__(self, proposal: Proposal) -> Mapping[str, Any]: ...


def score_recall(selected: list[str], gold: list[str]) -> float:
    return final_answer_recall(selected, gold)


def accept_external_score(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "quality",
        "per_question",
        "cost",
        "retention_ok",
        "protocol_ok",
        "projection_complete",
        "used_unavailable_info",
        "coverage",
        "delta_h",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"evaluator payload missing {missing}")
    return dict(payload)


def run_score(proposal: Proposal, fn: Callable[[Proposal], Mapping[str, Any]]) -> dict[str, Any]:
    return accept_external_score(fn(proposal))
