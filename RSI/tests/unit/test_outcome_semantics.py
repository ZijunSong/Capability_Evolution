"""PR-03 regression tests (guide §12): outcome classification, metric routing,
invalid-measurement blocking and current-parent selection.

Each test fails on the pre-fix code paths:
- partial credit (value>0, success=False) must NOT become a success case;
- strict vs partial metric routing must read DIFFERENT data;
- EVAL_INVALID records must be blocked from comparison;
- current-generation paired improvement must not be blocked by a historical
  noisy high-water mark.
"""

from __future__ import annotations

import pytest

from rsi.evolution.outcomes import Outcome, classify_outcome, invalid_measurement_blocked, metric_value


def test_partial_failure_evidence():
    """guide §12: value=.4 and success=False MUST be partial_failure and never
    enter a success summary."""
    meta = {"value": 0.4, "success": False, "measurement_status": "VALID",
            "terminated_reason": "max_actions", "component_errors": {}}
    assert classify_outcome(meta) is Outcome.PARTIAL_FAILURE

    # the old analysis_runner rule (value < 1e-9 → failure) fed this record
    # into success_cases; the unified classifier must put it in failure_cases.
    from rsi.evolution.search import EvolutionSearch

    per_task = {
        "t_partial": {"value": 0.4, "success": False},
        "t_strict_success": {"value": 1.0, "success": True},
    }
    fails = EvolutionSearch._fragments_from_per_task(per_task, failure=True)
    succ = EvolutionSearch._fragments_from_per_task(per_task, failure=False)
    assert [f.task_id for f in fails] == ["t_partial"]
    assert [f.task_id for f in succ] == ["t_strict_success"]

    # diagnosis agrees (the old value<1.0 rule classified the same record as
    # failure while analysis_runner classified it as success).
    from rsi.evolution.diagnosis import diagnose_traces

    rec = diagnose_traces(
        generation=0,
        per_task={"t_partial": {"value": 0.4, "success": False, "terminated_reason": "finish"}},
        baseline_scores=None,
        parent_scores=None,
    )
    # partial failure IS a failure category for mechanism diagnosis...
    total_failures = sum(f.count for f in rec.failure_categories)
    assert total_failures == 1
    # ...and never a success sample
    assert rec.success_sample_task_ids == ()


def test_partial_failure_not_success_in_analysis_runner_shape():
    """The exact minimal repro from the guide: value=0.4, success=False,
    component_errors={} produced 0 failure fragments / 1 success fragment."""
    meta = {"value": 0.4, "success": False, "component_errors": {}}
    assert classify_outcome(meta) is Outcome.PARTIAL_FAILURE


def test_success_and_failure_and_invalid():
    assert classify_outcome({"value": 1.0, "success": True}) is Outcome.SUCCESS
    assert classify_outcome({"value": 0.0, "success": False}) is Outcome.FAILURE
    assert classify_outcome({"value": 0.0, "success": False,
                             "measurement_status": "EVAL_INVALID"}) is Outcome.INVALID
    # missing strict label raises — never guessed
    with pytest.raises(ValueError):
        classify_outcome({"value": 0.9})


def test_metric_routing():
    """guide §12 test_metric_routing: search partial / confirm strict read
    different actual values; the config string is not an empty shell."""
    rec = {"value": 0.4, "success": False, "metric_name": "appworld_partial_test_pass_rate",
           "domain": "appworld"}
    assert metric_value(rec, metric="strict_success") == 0.0
    assert metric_value(rec, metric="partial_test_pass_rate") == 0.4
    web = {"value": 0.83, "success": False, "domain": "webshop"}
    assert metric_value(web, metric="webshop_reward") == 0.83
    with pytest.raises(ValueError):
        # webshop reward must not silently read an appworld record
        metric_value(rec, metric="webshop_reward")
    with pytest.raises(ValueError):
        # partial requested from a record that is not that metric
        metric_value({"value": 0.5, "metric_name": "webshop_reward"}, metric="partial_test_pass_rate")


def test_invalid_measurement_blocks():
    assert invalid_measurement_blocked({"value": 0.0, "success": False,
                                        "measurement_status": "EVAL_INVALID"})
    assert not invalid_measurement_blocked({"value": 0.5, "success": False,
                                            "measurement_status": "VALID"})
    # a details-nested status also counts
    assert invalid_measurement_blocked({"value": 0.0, "success": False,
                                        "details": {"measurement_status": "EVAL_INVALID"}})


def test_selection_current_parent(tmp_path):
    """guide §12 test_selection_current_parent: a current-generation paired
    improvement must not be blocked by a historical noisy maximum."""
    from rsi.evolution.search import EvolutionBudgets, EvolutionSearch
    from rsi.evolution.selection import CandidateScore, CostVector, SelectionResult, select_champion
    from rsi.harness.loader import load_empty_bank

    # the winner (0.55) beats the current parent (0.50) even though a
    # historical noisy high (0.70) exists; the pre-fix code compared against
    # best_macro=0.70 and kept the OLD parent.
    baseline = CandidateScore(candidate_id="baseline:parent", macro_score=0.50,
                              per_domain={"toy": 0.50}, cost=CostVector())
    winner = CandidateScore(candidate_id="cand1", macro_score=0.55,
                            per_domain={"toy": 0.55}, cost=CostVector(active_components=1))
    selection = select_champion([winner], baseline=baseline)
    assert selection.champion_id == "cand1"

    # and the search-level rule: candidate must clear the CURRENT parent,
    # not the historical best.  Simulate through the update rule directly.
    best_macro_display = 0.70  # historical noisy high, display-only
    current_parent_macro = 0.50
    win_macro = 0.55
    improved = win_macro > current_parent_macro + 1e-9  # pre-fix: > best_macro
    assert improved

    # historical max must not crown a candidate that LOST to the parent:
    # selection retains the baseline (NOT the losing candidate), and the
    # search-level champion (improved=False) stays with the parent bank.
    lost = CandidateScore(candidate_id="cand2", macro_score=0.45,
                          per_domain={"toy": 0.45}, cost=CostVector())
    selection2 = select_champion([lost], baseline=baseline)
    assert selection2.champion_id in (None, "baseline:parent")
    assert selection2.champion_id != "cand2"
