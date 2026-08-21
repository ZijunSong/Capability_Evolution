from __future__ import annotations

from typing import Any


def reward_delta(record: dict[str, Any]) -> float | None:
    before = record.get("reward_before")
    after = record.get("reward_after")
    if before is None or after is None:
        return None
    return float(after) - float(before)
