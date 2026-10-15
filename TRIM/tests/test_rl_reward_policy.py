"""Numeric checks for explicit TRIM rewards and the legacy wrapper."""

from __future__ import annotations

import math

import pytest

from trim.training.four_cell_runtime import terminal_reward_breakdown
from trim.training.reward_policy import score_explicit_reward


def _quality(**overrides):
    base = {
        "recall": 1.0,
        "precision": 1.0,
        "final_answer_recall": 1.0,
        "trajectory_recall": 1.0,
        "trajectory_fa_recall": 1.0,
        "n_curated": 1,
        "num_turns": 10,
    }
    base.update(overrides)
    return base


def test_final_answer_recall_is_the_metric_alone():
    scored = score_explicit_reward(_quality(final_answer_recall=0.5), mode="final-answer-recall", max_turns=40)
    assert scored["total"] == 0.5
    empty = score_explicit_reward(
        _quality(final_answer_recall=0.0, n_curated=0),
        mode="final-answer-recall",
        max_turns=40,
        terminal_format_error=True,
    )
    assert empty["total"] == 0.0


def test_composite_matches_v3_defaults_and_ignores_reward_version(monkeypatch):
    monkeypatch.setenv("REWARD_VERSION", "v4")
    scored = score_explicit_reward(_quality(num_turns=30), mode="harness1-composite", max_turns=40)
    assert scored["protocol"] == "harness1_v3_defaults_on_trim_metrics_v1"
    assert scored["total"] > 1.0
    assert math.isclose(
        scored["task"] + scored["penalty"] + scored["floor_adjustment"],
        scored["total"],
    )
    none = score_explicit_reward(_quality(n_curated=0), mode="harness1-composite", max_turns=40)
    assert none["total"] == -0.2
    fmt = score_explicit_reward(
        _quality(),
        mode="harness1-composite",
        max_turns=40,
        terminal_format_error=True,
    )
    assert fmt["total"] == -0.5
    floor = score_explicit_reward(
        _quality(recall=0.0, precision=0.0, final_answer_recall=0.0, trajectory_recall=0.0, trajectory_fa_recall=0.0),
        mode="harness1-composite",
        max_turns=40,
    )
    assert floor["total"] == pytest.approx(0.001)


def test_turn_penalty_uses_live_max_turns():
    wide = score_explicit_reward(_quality(num_turns=30), mode="harness1-composite", max_turns=40)
    narrow = score_explicit_reward(_quality(num_turns=30), mode="harness1-composite", max_turns=35)
    assert wide["turn_penalty"] < narrow["turn_penalty"]


def test_legacy_wrapper_is_unchanged_and_explicit_requires_fields():
    st = {"curated": {"g1": {}}, "pool": {"g1": {}}, "ended": True}
    actions = [{"name": "end_search", "arguments": {}}]
    legacy = terminal_reward_breakdown(
        st,
        query="q",
        gold_ids=["g1"],
        valids=[True],
        actions=actions,
    )
    assert legacy["protocol"] == "local_legacy_capped_shaping"
    assert legacy["total"] > 0
    with pytest.raises(ValueError):
        terminal_reward_breakdown(
            st,
            query="q",
            gold_ids=["g1"],
            valids=[True],
            actions=actions,
            rl_reward="final-answer-recall",
        )
    explicit = terminal_reward_breakdown(
        st,
        query="q",
        gold_ids=["g1"],
        valids=[True],
        actions=actions,
        row={"query": "q", "gold_docids": ["g1"]},
        rl_reward="final-answer-recall",
        max_turns=40,
        termination_kind="end_search",
    )
    assert explicit["total"] == 1.0
    with pytest.raises(ValueError):
        score_explicit_reward(_quality(recall=float("nan")), mode="final-answer-recall", max_turns=40)
