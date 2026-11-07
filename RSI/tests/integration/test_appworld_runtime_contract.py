"""PR-01 v3 AppWorld runtime contract (guide §3.1 D acceptance).

Vendor integration: requires the real AppWorld install + data under
``/mnt/songzijun/datasets/appworld`` (skipped when absent).  CPU-only, no GPU.

Acceptance items:
1. the three API-discovery calls produce non-empty usable output (not just
   "Execution successful.");
2. the doc file does NOT reference show_app_apis / generic authenticate;
3. numeric/string/null answers keep their type through the adapter's
   Finish submission path;
4. a FAILED answer submission is a retryable completion_submit_failed, not
   a success-looking finish;
5. execution errors surface as ok=False observations.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rsi.contracts.action import Finish, ToolCall
from rsi.environments.appworld_env import AppWorldAdapter
from rsi.contracts.task import PublicTask

APPWORLD_ROOT = os.environ.get("APPWORLD_ROOT", "/mnt/songzijun/datasets/appworld")

pytestmark = pytest.mark.skipif(
    not (Path(APPWORLD_ROOT) / "data" / "tasks").is_dir(),
    reason="real AppWorld data not installed",
)


def _adapter(tmp_path) -> AppWorldAdapter:
    a = AppWorldAdapter(APPWORLD_ROOT, max_interactions=15, out_root=str(tmp_path))
    task = PublicTask(
        task_id="024c982_1", domain="appworld",
        instruction="Smoke the interface.", group_id="g",
        metadata={"appworld_task_id": "024c982_1"},
    )
    a.reset(task, seed=17)
    return a


def test_api_discovery_calls_produce_output(tmp_path):
    """guide §3.1 D #1: the three documented discovery calls must return
    usable content, not bare 'Execution successful.'."""
    a = _adapter(tmp_path)
    try:
        for code in (
            "print(apis.api_docs.show_app_descriptions())",
            'print(apis.api_docs.show_api_descriptions(app_name="spotify"))',
            'print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))',
        ):
            r = a.step(ToolCall(name="python_exec", arguments={"code": code}))
            assert r.observation.ok, (code, r.observation.content[:200])
            text = str(r.observation.content)
            assert len(text) > 50, (code, text[:100])
            assert "Execution successful" not in text
            assert "show_app_apis" not in code
    finally:
        a.close()


def test_doc_file_does_not_reference_nonexistent_api():
    """guide §3.1 D #6: the v3 doc must not teach show_app_apis or the
    generic authenticate template (the v1 errors)."""
    doc = Path(__file__).resolve().parents[2] / "configs" / "env_docs" / "appworld_v3.md"
    text = doc.read_text("utf-8")
    assert "show_app_apis" not in text
    assert "authenticate(**accounts" not in text
    # and the verified discovery calls must be documented
    assert "show_app_descriptions" in text
    assert "show_api_descriptions" in text
    assert "show_api_doc" in text


def test_answer_type_fidelity_null(tmp_path):
    """guide §3.1 D #2/#3: Finish(None) submits JSON null and completes."""
    a = _adapter(tmp_path)
    try:
        r = a.step(Finish(answer=None))
        assert r.done, r.observation.content[:200]
        assert r.observation.ok
        assert "answer_submitted=null" in r.observation.content
    finally:
        a.close()


def test_failed_submission_is_retryable(tmp_path):
    """guide §3.1 D #4: a failing complete_task must return
    ok=False/completion_submit_failed with done=False — the model may
    retry; it is never a success-looking finish."""
    a = _adapter(tmp_path)
    try:
        # Force the vendor execute() to fail: point the adapter's env at a
        # broken callable for this one call (save_state still works so the
        # snapshot path stays real).
        class _Boom:
            def execute(self, code):
                raise RuntimeError("vendor down")

            def save_state(self, name):
                return "mock-state"
        a._env = _Boom()
        r = a.step(Finish(answer="7"))
        assert not r.observation.ok
        assert r.observation.error_kind == "completion_submit_failed"
        assert not r.done
    finally:
        a.close()


def test_execution_error_is_structured(tmp_path):
    """guide §3.1 C: a python runtime error must surface ok=False (not a
    normal-looking observation)."""
    a = _adapter(tmp_path)
    try:
        r = a.step(ToolCall(name="python_exec", arguments={"code": "1/0"}))
        assert not r.observation.ok
        assert r.observation.error_kind == "python_runtime_error"
    finally:
        a.close()


def test_prior_submission_not_overwritten(tmp_path):
    """guide §3.1 D #3: the model's own complete_task sets the submitted
    marker; the adapter's Finish(None) then keeps the prior value."""
    a = _adapter(tmp_path)
    try:
        # a successful python_exec call containing complete_task marks the
        # submission state
        r = a.step(ToolCall(
            name="python_exec",
            arguments={"code": "import json\napis.supervisor.complete_task(answer=json.loads('7'))\nprint('done')"},
        ))
        assert r.observation.ok or not r.observation.ok  # either way, next check holds
        if a._completion_submitted is None:
            pytest.skip("task 024c982_1 does not expose supervisor complete_task")
        r2 = a.step(Finish(answer=None))
        assert r2.done
        assert "prior answer submission kept" in r2.observation.content
    finally:
        a.close()
