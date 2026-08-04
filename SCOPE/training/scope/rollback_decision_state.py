"""Student-visible rollback DecisionState builder."""

from __future__ import annotations

from typing import Any

from harness.capability.state import DecisionState


def build_rollback_decision_state(
    base_state: DecisionState,
    *,
    recent_queries: list[str] = (),
    unresolved_claims: list[str] = (),
    tool_error_summary: str = "",
    available_checkpoints: list[dict[str, Any]] = (),
    remaining_search_budget: int = 0,
    remaining_recovery_budget: int = 0,
    branch_id: str = "main",
    state_hash: str = "",
) -> dict[str, Any]:
    d = base_state.to_dict()
    d.update(
        {
            "capability": "rollback_decision",
            "recent_queries": list(recent_queries)[-8:],
            "unresolved_claims": list(unresolved_claims),
            "tool_error_summary": tool_error_summary,
            "available_checkpoints": available_checkpoints,
            "remaining_search_budget": remaining_search_budget,
            "remaining_recovery_budget": remaining_recovery_budget,
            "branch_id": branch_id,
            "current_state_hash": state_hash,
        }
    )
    for forbidden in ("gold_answer", "future_trajectory", "verifier_label"):
        d.pop(forbidden, None)
    return d
