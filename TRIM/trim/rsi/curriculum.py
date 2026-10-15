"""Difficulty bands from work-split trajectories. A panel change resets patience."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


LEARNABLE_LOW = 0.1
LEARNABLE_HIGH = 0.7
TASK_FRONTIER_TOO_HARD = "TASK_FRONTIER_TOO_HARD"


def band_for_success(rate: float, *, low: float = LEARNABLE_LOW, high: float = LEARNABLE_HIGH) -> str:
    if rate < low:
        return "too_hard"
    if rate > high:
        return "too_easy"
    return "learnable"


def select_work_mix(
    records: Sequence[Mapping[str, Any]],
    *,
    current_fraction: float = 0.70,
    history_fraction: float = 0.30,
) -> dict[str, Any]:
    if abs(current_fraction + history_fraction - 1.0) > 1e-6:
        raise ValueError("curriculum fractions must sum to 1")
    learnable = [row for row in records if band_for_success(float(row["baseline_success"])) == "learnable"]
    history = [row for row in records if str(row.get("layer") or "") == "history"]
    current = [row for row in learnable if str(row.get("layer") or "current") == "current"]
    return {
        "current_ids": [str(row["query_id"]) for row in current],
        "history_ids": [str(row["query_id"]) for row in history],
        "current_fraction": current_fraction,
        "history_fraction": history_fraction,
        "n_learnable": len(learnable),
    }


def teacher_frontier(teacher_success: float, *, low: float = LEARNABLE_LOW) -> str | None:
    if teacher_success < low:
        return TASK_FRONTIER_TOO_HARD
    return None


def on_panel_change(previous_panel_id: str, next_panel_id: str) -> dict[str, Any]:
    changed = previous_panel_id != next_panel_id
    return {
        "panel_changed": changed,
        "reset_patience": changed,
        "rebaseline": changed,
    }
