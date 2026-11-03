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


@pytest.mark.parametrize("domain", ["appworld", "spreadsheetbench", "swebench", "webshop", "alfworld"])
def test_real_benchmark_official_parity_when_available(domain):
    """Adapter vs pinned official evaluator on a fixed action/fixture.

    PR-2 (audit §4.6/§14): a deps-present skip used to hide the entire real-
    benchmark parity gap.  Now:
      - deps absent -> explicit skip (no fallback environment is substituted);
      - appworld deps PRESENT -> the real no-op / known-correct / wrong-state
        controls in test_appworld_scorer_parity_controls.py run in this suite
        and FAIL on measurement bugs; a deps-present skip here is not allowed.
    """
    registry = global_registry()
    ok, missing = registry.check_deps(domain)
    if not ok:
        pytest.skip(f"{domain} optional deps missing: {list(missing)}; no fallback used")
    if domain == "appworld":
        # deps present: the real controls must run — re-execute them here so a
        # runner collecting only this file still gets the coverage.
        import importlib.util

        ctl_path = Path(__file__).parent / "test_appworld_scorer_parity_controls.py"
        spec = importlib.util.spec_from_file_location("appworld_parity_controls", ctl_path)
        controls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controls)
        controls.test_appworld_noop_control_initial_state(None)
        controls.test_appworld_scorer_does_not_initialize_a_new_world(None)
        controls.test_appworld_scorer_reads_actual_output_db(None)
        return
    # Other domains: importable deps but no wired controls yet.  This CI skip
    # is an explicit BLOCKED marker (audit §14) — the formal preflight gate
    # (rsi.cli doctor / formal runner) treats BLOCKED domains as non-formal.
    pytest.skip(
        f"BLOCKED: {domain} deps present but real parity controls not wired yet "
        "(audit doc §13 PR-2 scope); formal preflight must treat as non-formal"
    )
