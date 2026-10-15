"""Explicit terminal rewards. Legacy TRIM reward stays in four_cell_runtime.

``harness1-composite`` freezes Harness-1 v3 default coefficients on TRIM
quality metrics. It does not read ``REWARD_VERSION`` or other reward env vars.
``final-answer-recall`` is exactly the TRIM ``final_answer_recall`` metric.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

EXPLICIT_RL_REWARDS = ("harness1-composite", "final-answer-recall")
RL_REWARD_LEGACY = "trim-legacy-v1"
PROTOCOL_LEGACY = "local_legacy_capped_shaping"
PROTOCOL_HARNESS1_V3 = "harness1_v3_defaults_on_trim_metrics_v1"
PROTOCOL_FA_ONLY = "trim_final_answer_recall_only_v1"
TERMINATION_KINDS = frozenset(
    {
        "end_search",
        "max_turns",
        "format_error",
        "execution_error",
        "infrastructure_error",
    }
)

_REWARD_PROTOCOL = {
    RL_REWARD_LEGACY: PROTOCOL_LEGACY,
    "harness1-composite": PROTOCOL_HARNESS1_V3,
    "final-answer-recall": PROTOCOL_FA_ONLY,
}


def reward_protocol_for(mode: str | None) -> str:
    key = str(mode or RL_REWARD_LEGACY)
    if key not in _REWARD_PROTOCOL:
        raise ValueError(f"unknown rl_reward={mode!r}")
    return _REWARD_PROTOCOL[key]


def classify_termination(
    st: Mapping[str, Any] | None,
    *,
    names: Sequence[str] | None = None,
    max_turns: int,
) -> str:
    """Map an episode end onto the five diagnostic reasons.

    A recorded ``termination_kind`` wins. Upstream format failures set that
    field at the invalid-action boundary; a later successful retry is not
    labeled here unless the episode actually ended on the format error.
    """
    state = dict(st or {})
    explicit = str(state.get("termination_kind") or "")
    if explicit in TERMINATION_KINDS:
        return explicit
    if bool(state.get("terminal_format_error")):
        return "format_error"
    reason = str(state.get("end_reason") or "").lower()
    if "format" in reason:
        return "format_error"
    if "infra" in reason:
        return "infrastructure_error"
    if "exec" in reason:
        return "execution_error"
    tool_names = [str(x) for x in (names or [])]
    if state.get("ended"):
        if tool_names and tool_names[-1] == "end_search":
            return "end_search"
        return "end_search"
    if len(tool_names) >= int(max_turns):
        return "max_turns"
    return "execution_error"


def score_explicit_reward(
    quality: Mapping[str, Any],
    *,
    mode: str,
    max_turns: int,
    terminal_format_error: bool = False,
) -> dict[str, Any]:
    allowed = set(EXPLICIT_RL_REWARDS)
    if mode not in allowed:
        raise ValueError(f"unknown rl_reward={mode!r}")

    names = (
        "recall",
        "precision",
        "final_answer_recall",
        "trajectory_recall",
        "trajectory_fa_recall",
    )
    try:
        values = {key: float(quality[key]) for key in names}
    except KeyError as exc:
        raise ValueError(f"reward metric missing: {exc}") from exc
    if any(not isfinite(x) or x < 0.0 or x > 1.0 for x in values.values()):
        raise ValueError("reward metrics must be finite and in [0, 1]")
    fa = values["final_answer_recall"]
    result: dict[str, Any] = {
        "rl_reward": mode,
        "task_recall": fa,
        "final_answer_recall": fa,
        "legal": 0.0,
        "shaping": 0.0,
        "penalty": 0.0,
        "floor_adjustment": 0.0,
        "metrics": values,
    }

    if mode == "final-answer-recall":
        return {
            **result,
            "total": fa,
            "task": fa,
            "protocol": PROTOCOL_FA_ONLY,
        }

    result["protocol"] = PROTOCOL_HARNESS1_V3
    if terminal_format_error:
        return {**result, "total": -0.5, "task": 0.0, "penalty": -0.5}
    try:
        n_curated = int(quality["n_curated"])
    except KeyError as exc:
        raise ValueError("reward metric missing: 'n_curated'") from exc
    if n_curated == 0:
        return {**result, "total": -0.2, "task": 0.0, "penalty": -0.2}

    recall, precision = values["recall"], values["precision"]
    f2 = 0.0 if recall + precision == 0.0 else (5.0 * precision * recall / (4.0 * precision + recall))
    traj = values["trajectory_recall"]
    traj_fa = values["trajectory_fa_recall"]
    try:
        turn = int(quality["num_turns"])
    except KeyError as exc:
        raise ValueError("reward metric missing: 'num_turns'") from exc
    if int(max_turns) <= 0:
        raise ValueError("max_turns must be positive")
    turn_penalty = 0.15 * min(max(turn - 24, 0) / max(int(max_turns) - 24, 1), 1.0)
    fa_miss_penalty = 0.35 * max(traj_fa - fa, 0.0)
    terms = {
        "outcome_f2": 0.7 * f2,
        "trajectory_recall_term": 0.3 * traj,
        "final_answer_bonus": 1.0 if fa > 0.0 else 0.0,
        "final_answer_dense": 0.8 * fa,
        "trajectory_fa_dense": 0.4 * traj_fa,
    }
    positive = sum(terms.values())
    penalty = -(fa_miss_penalty + turn_penalty)
    unclipped = positive + penalty
    total = max(0.001, unclipped)
    return {
        **result,
        **terms,
        "total": total,
        "task": positive,
        "penalty": penalty,
        "floor_adjustment": total - unclipped,
        "f_beta": f2,
        "turn_penalty": turn_penalty,
        "fa_miss_penalty": fa_miss_penalty,
    }
