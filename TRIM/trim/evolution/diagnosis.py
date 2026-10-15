"""Failure clusters. The raw trajectory id stays attached to the short label."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

FAILURE_LABELS = (
    "bad_query",
    "unread_evidence",
    "redundant_search",
    "bad_selection",
    "early_stop",
    "context_overflow",
    "format_or_tool_error",
    "solved",
)


def label_trajectory(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("failure") or "")
    if explicit in FAILURE_LABELS:
        return explicit
    if float(row.get("recall") or 0.0) >= 1.0:
        return "solved"
    if row.get("format_error"):
        return "format_or_tool_error"
    return "bad_query"


def cluster_failures(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = label_trajectory(row)
        grouped[label].append(
            {
                "query_id": str(row.get("query_id") or ""),
                "trajectory_id": str(row.get("trajectory_id") or ""),
                "recall": float(row.get("recall") or 0.0),
                "failure": label,
            }
        )
    return dict(grouped)
