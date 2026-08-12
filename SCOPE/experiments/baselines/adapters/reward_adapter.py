"""Map SCOPE episode outcomes to baseline reward schemas."""

from __future__ import annotations

from typing import Any


def scope_to_scalar_reward(episode: dict[str, Any]) -> float:
    for key in ("reward", "recall", "final_answer_recall", "score"):
        if key in episode and episode[key] is not None:
            return float(episode[key])
    raise KeyError("episode missing reward/recall/score — no silent zero")


def map_rewards(episodes: list[dict[str, Any]], *, schema: str = "scalar") -> list[dict[str, Any]]:
    out = []
    for ep in episodes:
        r = scope_to_scalar_reward(ep)
        if schema == "scalar":
            out.append({"query_id": ep.get("query_id"), "reward": r})
        elif schema == "grpo":
            out.append({"query_id": ep.get("query_id"), "scores": [r], "advantages": [r]})
        else:
            raise ValueError(f"unknown reward schema: {schema}")
    return out
