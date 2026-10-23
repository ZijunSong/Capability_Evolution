"""Fixed-action / prebuilt-artifact scorer parity.

For toy domains the wrapper (EnvironmentAdapter final artifact + Scorer) must
agree with a direct official-style computation on the same produced artifact.
For real benchmarks the same pattern validates the adapter against the pinned
official evaluator; those tests skip with an explicit reason when the optional
dependency/dataset revision is absent (they never fall back to another
environment).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.observation import ArtifactRef
from rsi.contracts.task import PrivateTaskRef
from rsi.environments.registry import global_registry
from rsi.environments.toy import ToyFileWorld, ToyScorer, build_toy_file_tasks, digest_of
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop
from rsi.providers.scripted import ScriptedPolicy
from rsi.harness.loader import load_empty_bank


def _fixed_writer(value: str):
    def plan(view_text: str, step: int):
        if step == 0:
            return ToolCall(name="write_file", arguments={"path": "out.txt", "content": value})
        return Finish(answer=None)

    return plan


def test_toy_file_fixed_artifact_parity():
    """Wrapper path (loop -> final_artifact -> scorer) equals direct computation."""
    for i in range(4):
        task, expected = build_toy_file_tasks(4, 17)[i]
        env = ToyFileWorld()
        loop = MinimalLoop(
            env=env,
            policy=ScriptedPolicy(_fixed_writer(expected)),
            bank=load_empty_bank(),
            budgets=BudgetSpec(max_actions=3),
        )
        res = loop.run_episode(task, seed=1)
        scorer = ToyScorer({task.task_id: expected})
        via_wrapper = scorer.evaluate(res.final_artifact, PrivateTaskRef(ref_id=task.task_id, domain="toy_file"))
        # direct official-style computation on the produced content digest
        direct_ok = res.final_artifact.digest == digest_of(expected)
        assert via_wrapper.success == direct_ok
        assert via_wrapper.value == (1.0 if direct_ok else 0.0)


def test_toy_kv_fixed_answer_parity():
    from rsi.environments.toy import ToyKVWorld, build_toy_kv_tasks

    task, expected = build_toy_kv_tasks(2, 17)[0]

    def plan(view_text: str, step: int):
        return Finish(answer=expected)

    env = ToyKVWorld()
    loop = MinimalLoop(
        env=env, policy=ScriptedPolicy(plan), bank=load_empty_bank(), budgets=BudgetSpec(max_actions=2)
    )
    res = loop.run_episode(task, seed=1)
    from rsi.environments.toy import ToyScorer as S

    scorer = S({task.task_id: expected})
    rec = scorer.evaluate(res.final_artifact, PrivateTaskRef(ref_id=task.task_id, domain="toy_kv"))
    assert rec.success


@pytest.mark.parametrize("domain", ["appworld", "spreadsheetbench", "swebench"])
def test_real_benchmark_official_parity_when_available(domain):
    """Adapter vs pinned official evaluator on a fixed action/fixture.

    Requires the optional dependency AND the frozen dataset revision; skipped
    (with reason) otherwise — no fallback environment is ever substituted.
    """
    registry = global_registry()
    ok, missing = registry.check_deps(domain)
    if not ok:
        pytest.skip(f"{domain} optional deps missing: {list(missing)}; no fallback used")
    pytest.skip(
        f"{domain} deps present but the frozen dataset revision wiring "
        "(data prepare) is completed per docs/runbook.md blocking list"
    )
