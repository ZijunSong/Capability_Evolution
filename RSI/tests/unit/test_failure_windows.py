"""PR-04 tests (guide §12 / §4.3): failure-window packet + repair budget."""

from __future__ import annotations

from rsi.evolution.failure_windows import (
    failure_windows_from_steps,
    hypothesis_quality_problems,
)


def test_partial_failure_gets_windows_not_success():
    """value=0.4, success=False is a failure — its steps produce windows."""
    steps = [
        {"order": 0, "kind": "tool_call", "action": "print(x)", "obs_head": "ok"},
        {"order": 1, "kind": "tool_call", "action": "apis.foo.bar()",
         "obs_head": "Execution failed. Traceback: name 'apis.foo' is not defined"},
    ]
    ws = failure_windows_from_steps(
        task_id="t", steps=steps, terminated_reason="max_actions",
        value=0.4, success=False,
    )
    assert ws, "partial failure must produce evidence windows"
    assert any(w.failure_type == "api_error" for w in ws)
    # the window pairs the error with its predecessor step
    assert "prev[" in ws[0].evidence_text or "Execution failed" in ws[0].evidence_text


def test_success_and_invalid_produce_nothing():
    assert failure_windows_from_steps(
        task_id="t", steps=[{"order": 0, "action": "a", "obs_head": "b"}],
        terminated_reason="finish", value=1.0, success=True,
    ) == ()
    assert failure_windows_from_steps(
        task_id="t", steps=[], terminated_reason="finish", value=0.0, success=False,
        measurement_status="EVAL_INVALID",
    ) == ()


def test_budget_termination_gets_a_window():
    ws = failure_windows_from_steps(
        task_id="t", steps=[{"order": 0, "action": "look", "obs_head": "you see a room"}],
        terminated_reason="budget_wall", value=0.0, success=False,
    )
    assert ws and ws[0].failure_type == "budget"


def test_hypothesis_quality_contract():
    # generic advice + no evidence → rejected
    assert hypothesis_quality_problems("be more careful when answering", ())
    # mechanism + evidence → accepted
    ok = hypothesis_quality_problems(
        "mechanism: guard against unknown api names; triggers when show_app_apis "
        "is called; expected_change: correct discovery call; possible_regression: none",
        ("t1:3", "t2:0"),
    )
    assert ok == ()
    # mechanism but no evidence → rejected
    assert any("evidence" in p for p in hypothesis_quality_problems(
        "mechanism: cache api docs between turns to avoid re-querying them each step", ()
    ))
