"""Argmax / threshold decision for rollback typed operations."""

from __future__ import annotations

from collections.abc import Iterable
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


_TIE_ORDER = (
    RollbackOperation.CONTINUE,
    RollbackOperation.REPLAN,
    RollbackOperation.ROLLBACK_TO,
)

# Round-9/10 P0: train has no REPLAN labels; keep REPLAN out of the decision space.
DEFAULT_ALLOWED_OPERATIONS: tuple[RollbackOperation, ...] = (
    RollbackOperation.CONTINUE,
    RollbackOperation.ROLLBACK_TO,
)


def decide_rollback_operation(
    *,
    score_continue: float,
    score_replan: float,
    score_rollback: float,
    threshold: float = 0.0,
    candidate_checkpoint_id: str | None = None,
    tie_eps: float = 1e-5,
    allowed_operations: Iterable[RollbackOperation] | None = None,
    disable_replan: bool = False,
    near_boundary_prefer_continue_eps: float = 0.0,
) -> RollbackDecision:
    """Decide CONTINUE/REPLAN/ROLLBACK_TO from verbalizer scores.

    Round-10 parity contract: when ``disable_replan`` and
    ``near_boundary_prefer_continue_eps>0``, prefer CONTINUE whenever
    ``|score_continue-score_rollback| <= eps``. This stable tie rule is applied
    identically on HF and vLLM to eliminate near-boundary backend flips.
    """
    scores = {
        RollbackOperation.CONTINUE: float(score_continue),
        RollbackOperation.REPLAN: float(score_replan),
        RollbackOperation.ROLLBACK_TO: float(score_rollback),
    }
    if allowed_operations is None and disable_replan:
        allowed = set(DEFAULT_ALLOWED_OPERATIONS)
    elif allowed_operations is None:
        allowed = set(_TIE_ORDER)
    else:
        allowed = set(allowed_operations)
    if not allowed:
        raise ValueError("allowed_operations must be non-empty")
    active = {op: scores[op] for op in _TIE_ORDER if op in allowed}
    best_score = max(active.values())
    # Stable tie-break: prefer CONTINUE > REPLAN > ROLLBACK_TO on exact/near-exact ties.
    # Numerical HF↔vLLM parity relies on float32 HF scoring + token-id vLLM prompts;
    # keep tie_eps tight so clear margins are not collapsed.
    tied = [op for op in _TIE_ORDER if op in active and abs(active[op] - best_score) <= tie_eps]
    best_op = tied[0]
    ordered = sorted(active.values(), reverse=True)
    second = ordered[1] if len(ordered) > 1 else ordered[0]
    margin = best_score - second

    if best_op == RollbackOperation.ROLLBACK_TO and margin < threshold:
        best_op = RollbackOperation.CONTINUE

    # Binary near-boundary stabilize (Round 10 Gate A): force CONTINUE in the deadzone.
    if (
        disable_replan
        and near_boundary_prefer_continue_eps > 0
        and RollbackOperation.CONTINUE in active
        and RollbackOperation.ROLLBACK_TO in active
    ):
        signed = scores[RollbackOperation.CONTINUE] - scores[RollbackOperation.ROLLBACK_TO]
        if abs(signed) <= near_boundary_prefer_continue_eps:
            best_op = RollbackOperation.CONTINUE
            margin = abs(signed)

    ck = candidate_checkpoint_id if best_op == RollbackOperation.ROLLBACK_TO else None
    return RollbackDecision(
        predicted_operation=best_op,
        checkpoint_id=ck,
        score_continue=scores[RollbackOperation.CONTINUE],
        score_replan=scores[RollbackOperation.REPLAN],
        score_rollback=scores[RollbackOperation.ROLLBACK_TO],
        margin=margin,
    )
