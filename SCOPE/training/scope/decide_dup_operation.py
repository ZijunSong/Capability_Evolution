"""Single pure decision function for KEEP/SKIP admission (Round 7)."""

from __future__ import annotations

from dataclasses import dataclass

from harness.capability.dup_operation import DupOperation


COMPARISON_OPERATOR = ">="


@dataclass(frozen=True)
class DupDecisionResult:
    score_keep: float
    score_skip: float
    margin: float
    threshold: float
    comparison_operator: str
    predicted_operation: DupOperation

    @property
    def margin_definition(self) -> str:
        return "score_skip-score_keep"


def decide_dup_operation(
    *,
    score_keep: float,
    score_skip: float,
    threshold: float,
) -> DupDecisionResult:
    """margin = score_skip - score_keep; SKIP iff margin >= threshold."""
    margin = score_skip - score_keep
    operation = (
        DupOperation.SKIP_DUPLICATE
        if margin >= threshold
        else DupOperation.KEEP_EVIDENCE
    )
    return DupDecisionResult(
        score_keep=score_keep,
        score_skip=score_skip,
        margin=margin,
        threshold=threshold,
        comparison_operator=COMPARISON_OPERATOR,
        predicted_operation=operation,
    )
