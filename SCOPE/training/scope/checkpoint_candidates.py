"""Shared checkpoint candidate ordering and local ID mapping for rollback."""

from __future__ import annotations

from typing import Any


def _sort_key(ck: dict[str, Any]) -> tuple:
    return (
        -int(ck.get("turn_id", 0)),
        -int(ck.get("n_curated", 0)),
        -int(ck.get("n_pool", 0)),
        str(ck.get("checkpoint_id", "")),
    )


def order_checkpoint_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic ordering with stable tie-break on checkpoint_id."""
    return sorted(candidates, key=_sort_key)


def assign_local_checkpoint_ids(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    ordered = order_checkpoint_candidates(candidates)
    local_to_global: dict[str, str] = {}
    enriched: list[dict[str, Any]] = []
    for idx, ck in enumerate(ordered):
        local_id = f"C{idx}"
        global_id = str(ck.get("checkpoint_id", ""))
        local_to_global[local_id] = global_id
        enriched.append(
            {
                **ck,
                "local_checkpoint_id": local_id,
                "relative_turn": int(ck.get("turn_id", 0)),
                "evidence_count": int(ck.get("n_curated", 0)),
                "verified_count": int(ck.get("n_verified", ck.get("n_curated", 0))),
            }
        )
    return enriched, local_to_global


def global_to_local_id(global_id: str | None, local_to_global: dict[str, str]) -> str | None:
    if not global_id:
        return None
    for local, global_ck in local_to_global.items():
        if global_ck == global_id:
            return local
    return None


def summarize_candidate(ck: dict[str, Any]) -> str:
    return (
        f"{ck.get('local_checkpoint_id', '?')}:"
        f"turn={ck.get('relative_turn', ck.get('turn_id', 0))},"
        f"evidence={ck.get('evidence_count', ck.get('n_curated', 0))},"
        f"verified={ck.get('verified_count', 0)},"
        f"budget_rem={ck.get('remaining_recovery_budget', '?')}"
    )
