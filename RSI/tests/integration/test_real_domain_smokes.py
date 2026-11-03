"""Real-domain readiness smokes (task-book benchmark domains).

Each smoke runs the adapter + official-scorer chain end-to-end on one real
task and is SKIPPED with a precise reason when the domain's data/deps are not
ready.  These gate the evolve submissions: a domain that cannot smoke cannot
legitimately contribute to the harness panels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.contracts.action import ToolCall
from rsi.environments.registry import global_registry


def _probe(domain: str):
    return global_registry().probe_executable(domain)


def test_appworld_smoke():
    ok, why = _probe("appworld")
    if not ok:
        pytest.skip(f"appworld not ready: {why}")
    registry = global_registry()
    adapter = registry.get("appworld").adapter_factory()
    provider = registry.get("appworld").task_provider_factory()
    scorer = registry.get("appworld").scorer_factory()
    tasks = provider.public_tasks("work_train")
    assert tasks, "no work_train tasks"
    task = tasks[0]
    obs = adapter.reset(task, seed=1)
    assert "Task:" in obs.content or "task" in obs.content.lower()
    res = adapter.step(_py("1+1"))
    assert res.observation.ok, res.observation.content[:200]
    snap = adapter.snapshot()
    fork = adapter.fork(snap)
    assert fork.state_digest()  # fork is a live world
    fork.close()
    art = adapter.final_artifact()
    rec = scorer.evaluate(art, provider.private_ref(task.task_id))
    assert 0.0 <= rec.value <= 1.0 and rec.metric_name == "task_goal_completion"
    assert not rec.synthetic
    adapter.close()


def _py(code: str):
    from rsi.contracts.action import ToolCall

    return ToolCall(name="python_exec", arguments={"code": code})


def test_spreadsheetbench_smoke():
    ok, why = _probe("spreadsheetbench")
    if not ok:
        pytest.skip(f"spreadsheetbench not ready: {why}")
    registry = global_registry()
    adapter = registry.get("spreadsheetbench").adapter_factory()
    provider = registry.get("spreadsheetbench").task_provider_factory()
    scorer = registry.get("spreadsheetbench").scorer_factory()
    task = provider.public_tasks("work_train")[0]
    obs = adapter.reset(task, seed=1)
    assert "input" in obs.content.lower()
    art = adapter.final_artifact()
    rec = scorer.evaluate(art, provider.private_ref(task.task_id))
    assert rec.metric_name == "execution_correctness"
    assert not rec.synthetic
    adapter.close()


def test_swebench_smoke():
    ok, why = _probe("swebench")
    if not ok:
        pytest.skip(f"swebench not ready: {why}")
    from rsi.contracts.errors import EnvironmentError_

    registry = global_registry()
    adapter = registry.get("swebench").adapter_factory()
    provider = registry.get("swebench").task_provider_factory()
    task = provider.public_tasks("work_train")[0]
    try:
        obs = adapter.reset(task, seed=1)
    except EnvironmentError_ as e:
        pytest.skip(f"swebench workspace not prepared: {e}")
    assert "workspace ready" in obs.content.lower()
    res = adapter.step(_sh("git status --short"))
    assert res.observation.ok or "not a git" in res.observation.content
    art = adapter.final_artifact()
    assert art.digest
    adapter.close()


def test_webshop_smoke():
    ok, why = _probe("webshop")
    if not ok:
        pytest.skip(f"webshop not ready: {why}")
    from rsi.contracts.errors import NotImplementedCapability

    registry = global_registry()
    adapter = registry.get("webshop").adapter_factory()
    provider = registry.get("webshop").task_provider_factory()
    scorer = registry.get("webshop").scorer_factory()
    tasks = provider.public_tasks("work_train")
    assert tasks, "no work_train tasks"
    task = tasks[0]
    try:
        obs = adapter.reset(task, seed=1)
    except NotImplementedCapability as e:
        pytest.skip(f"webshop data not prepared: {e}")
    assert "Task:" in obs.content
    res = adapter.step(ToolCall(name="search", arguments={"query": task.instruction[:40]}))
    assert res.observation.ok, res.observation.content[:200]
    snap = adapter.snapshot()
    fork = adapter.fork(snap)
    assert fork.state_digest()
    fork.close()
    art = adapter.final_artifact()
    scorer.load_private(task.task_id, provider.private_record(task.task_id))
    rec = scorer.evaluate(art, provider.private_ref(task.task_id))
    assert rec.metric_name == "task_reward" and 0.0 <= rec.value <= 1.0
    assert not rec.synthetic
    adapter.close()


def test_alfworld_smoke():
    ok, why = _probe("alfworld")
    if not ok:
        pytest.skip(f"alfworld not ready: {why}")
    from rsi.contracts.errors import EnvironmentError_

    registry = global_registry()
    adapter = registry.get("alfworld").adapter_factory()
    provider = registry.get("alfworld").task_provider_factory()
    scorer = registry.get("alfworld").scorer_factory()
    tasks = provider.public_tasks("work_train")
    assert tasks, "no work_train tasks"
    task = tasks[0]
    try:
        obs = adapter.reset(task, seed=1)
    except EnvironmentError_ as e:
        pytest.skip(f"alfworld game not prepared: {e}")
    assert obs.content
    res = adapter.step(ToolCall(name="command", arguments={"text": "look"}))
    assert res.observation.ok, res.observation.content[:200]
    snap = adapter.snapshot()
    fork = adapter.fork(snap)
    assert fork.state_digest()
    fork.close()
    art = adapter.final_artifact()
    rec = scorer.evaluate(art, provider.private_ref(task.task_id))
    assert rec.metric_name == "task_success" and rec.value in (0.0, 1.0)
    assert not rec.synthetic
    adapter.close()


def _sh(cmd: str):
    from rsi.contracts.action import ToolCall

    return ToolCall(name="shell_exec", arguments={"cmd": cmd})
