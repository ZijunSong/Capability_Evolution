"""Selection baseline survival + diagnosis honesty (audit doc §8.1, §5.3).

Pins the PR-5 fixes:

1. the doc's isolated repro: H0=0.8, sole candidate=0.1 with an allowed
   domain regression of 0.05 — the old code crowned the candidate; the
   incumbent must now be retained (NO_SUPPORTED_IMPROVEMENT);
2. best_macro starts at the parent's measured macro, so a first-generation
   candidate does not auto-become champion;
3. budget_wall diagnosis no longer maps to the canned "compress / finish
   earlier" mechanism hint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi.evolution.diagnosis import _mechanism_hint, diagnose_traces
from rsi.evolution.selection import CandidateScore, CostVector, select_champion


def _score(cid: str, macro: float, per_domain: dict[str, float]) -> CandidateScore:
    return CandidateScore(
        candidate_id=cid,
        macro_score=macro,
        per_domain=per_domain,
        cost=CostVector(active_components=1, latency_ms=1000.0),
    )


def test_incumbent_survives_when_all_candidates_regress():
    """The audit doc's isolated repro (§3, last paragraph)."""
    baseline = _score("H0", 0.8, {"appworld": 0.8})
    cand = _score("cand", 0.1, {"appworld": 0.05})  # regression far beyond 0.05
    result = select_champion([cand], baseline=baseline, tolerance=0.01, max_domain_regression=0.05)
    assert result.champion_id == "H0", result.to_json()
    assert "NO_SUPPORTED_IMPROVEMENT" in result.reason


def test_incumbent_survives_when_all_candidates_invalid():
    baseline = _score("H0", 0.8, {"appworld": 0.8})
    bad = CandidateScore(
        candidate_id="bad",
        macro_score=0.9,
        per_domain={"appworld": 0.9},
        cost=CostVector(),
        protocol_ok=False,
    )
    result = select_champion([bad], baseline=baseline)
    assert result.champion_id == "H0"
    assert "NO_SUPPORTED_IMPROVEMENT" in result.reason


def test_real_improvement_still_wins():
    baseline = _score("H0", 0.8, {"appworld": 0.8})
    cand = _score("cand", 0.9, {"appworld": 0.9})
    result = select_champion([cand], baseline=baseline)
    assert result.champion_id == "cand"


def test_baseline_never_dropped_by_domain_regression_filter():
    """Even a baseline whose own score is below `max_domain_regression` vs an
    empty reference must remain in the pool — the filter applies to
    candidates only."""
    baseline = _score("H0", 0.03, {"appworld": 0.03})
    cand = _score("cand", 0.02, {"appworld": 0.02})
    result = select_champion([cand], baseline=baseline, max_domain_regression=0.05)
    assert result.champion_id in {"H0", "cand"}
    # H0 must be in the ordered pool
    assert "H0" in result.ordered


def test_budget_wall_hint_does_not_prescribe_compression():
    hint = _mechanism_hint("budget_wall")
    assert "UNKNOWN_NEEDS_TRACE" in hint
    assert "compress" not in hint.lower().split("not justified")[0] or (
        "is NOT justified" in hint
    )
    # the old canned strategies must not be recommended
    assert "consider context compression" not in hint
    assert "earlier finishing" not in hint


def test_diagnosis_carries_unknown_needs_trace_for_budget_wall():
    rec = diagnose_traces(
        generation=0,
        per_task={
            "t1": {"value": 0.0, "success": False, "terminated_reason": "budget_wall", "n_decisions": 0},
            "t2": {"value": 0.5, "success": False, "terminated_reason": "finish", "n_decisions": 12},
        },
        baseline_scores=None,
        parent_scores=None,
    )
    wall = [f for f in rec.failure_categories if f.kind == "budget_wall"]
    assert wall, rec.to_json()
    assert "UNKNOWN_NEEDS_TRACE" in wall[0].mechanism_summary
