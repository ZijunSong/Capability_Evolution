"""Argmax / threshold decision for rollback typed operations."""

from __future__ import annotations

from dataclasses import dataclass

from harness.capability.rollback_operation import RollbackOperation


@dataclass
class RollbackDecision:
    predicted_operation: RollbackOperation
    checkpoint_id: str | None
    score_continue: float = 0.0
    score_replan: float = 0.0
    score_rollback: float = 0.0
    margin: float = 0.0


def decide_rollback_operation(
    *,
    score_continue: float,
    score_replan: float,
    score_rollback: float,
    threshold: float = 0.0,
    candidate_checkpoint_id: str | None = None,
) -> RollbackDecision:
    scores = {
        RollbackOperation.CONTINUE: score_continue,
        RollbackOperation.REPLAN: score_replan,
        RollbackOperation.ROLLBACK_TO: score_rollback,
    }
    best_op = max(scores, key=scores.get)
    best_score = scores[best_op]
    second = sorted(scores.values(), reverse=True)[1]
    margin = best_score - second

    if best_op == RollbackOperation.ROLLBACK_TO and margin < threshold:
        best_op = RollbackOperation.CONTINUE

    ck = candidate_checkpoint_id if best_op == RollbackOperation.ROLLBACK_TO else None
    return RollbackDecision(
        predicted_operation=best_op,
        checkpoint_id=ck,
        score_continue=score_continue,
        score_replan=score_replan,
        score_rollback=score_rollback,
        margin=margin,
    )
