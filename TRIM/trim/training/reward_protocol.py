"""Single terminal-score entry for explicit RSI rewards.

Bank code cannot call this. The trainer passes an episode and labels in.
"""

from __future__ import annotations

from typing import Any, Mapping

from trim.eval.harness1_metrics import episode_quality_metrics
from trim.rsi.metrics import final_answer_recall
from trim.training.reward_policy import (
    PROTOCOL_FA_ONLY,
    PROTOCOL_HARNESS1_V3,
    score_explicit_reward,
)


def score_terminal(protocol: str, episode: Mapping[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    mode = _mode(protocol)
    selected = list(episode.get("selected_doc_ids") or [])
    gold = [str(x) for x in (labels.get("gold_doc_ids") or labels.get("gold_docids") or [])]
    if mode == "final-answer-recall" and selected:
        fa = final_answer_recall(selected, gold)
        return {
            "rl_reward": mode,
            "protocol": PROTOCOL_FA_ONLY,
            "total": fa,
            "task": fa,
            "task_recall": fa,
            "final_answer_recall": fa,
            "legal": 0.0,
            "shaping": 0.0,
            "penalty": 0.0,
        }
    row = dict(labels.get("row") or {})
    if gold and "gold_docids" not in row:
        row = {**row, "gold_docids": gold}
    st = dict(episode.get("state") or {})
    actions = list(episode.get("actions") or [])
    quality = episode_quality_metrics(
        st,
        row,
        tool_names=[str(action.get("name") or "") for action in actions],
        valids=list(episode.get("valids") or []),
        reward=0.0,
        max_turns=int(episode["max_turns"]),
        actions=actions,
    )
    if mode == "final-answer-recall":
        scored = score_explicit_reward(
            quality,
            mode=mode,
            max_turns=int(episode["max_turns"]),
            terminal_format_error=False,
        )
        return scored
    return score_explicit_reward(
        quality,
        mode=mode,
        max_turns=int(episode["max_turns"]),
        terminal_format_error=str(episode.get("termination_kind") or "") == "format_error",
    )


def _mode(protocol: str) -> str:
    if protocol in {PROTOCOL_FA_ONLY, "final-answer-recall"}:
        return "final-answer-recall"
    if protocol in {PROTOCOL_HARNESS1_V3, "harness1-composite"}:
        return "harness1-composite"
    raise ValueError(f"unknown reward protocol {protocol!r}")
