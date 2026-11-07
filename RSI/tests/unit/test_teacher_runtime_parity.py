"""PR-07 tests (guide §12): teacher/runtime parity — veto feedback, finish
hooks and budget accounting inside the teacher branch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.events import Visibility
from rsi.environments.toy import ToyFileWorld, build_toy_file_tasks
from rsi.harness.loader import load_empty_bank
from rsi.opd.teacher import TeacherForkRunner
from rsi.runtime.budgets import BudgetSpec
from rsi.providers.scripted import ScriptedPolicy


class FinisherPlan:
    """Always proposes Finish — teacher must pass before_finish gating."""

    def __call__(self, view_text: str, step: int):
        return Finish(answer="done")


class BlockedFinishBank:
    """A bank whose finish-guard always blocks (simulating a finish-guard
    component so the teacher's Finish must NOT terminate the branch)."""

    def __init__(self, bank):
        self._bank = bank

    def __getattr__(self, name):
        return getattr(self._bank, name)

    @property
    def components(self):
        return self._bank.components


def _veto_bank_source() -> str:
    return (
        "from rsi.harness.component import ProposalEffect, StatePatch, ViewPatch, FinishEffect\n"
        "class AlwaysVeto:\n"
        "    id = 'always_veto'\n"
        "    def initialize(self, ctx):\n        return StatePatch()\n"
        "    def before_model(self, view, api):\n        return ViewPatch(changed=False)\n"
        "    def after_proposal(self, action, view, api):\n"
        "        return ProposalEffect(veto=True, veto_reason='test veto')\n"
        "    def after_observation(self, event, api):\n        return StatePatch()\n"
        "    def before_finish(self, action, view, api):\n        return FinishEffect()\n"
        "component = AlwaysVeto\n"
    )


def _finish_block_bank_source() -> str:
    return (
        "from rsi.harness.component import ProposalEffect, StatePatch, ViewPatch, FinishEffect\n"
        "class BlockFinish:\n"
        "    id = 'block_finish'\n"
        "    def initialize(self, ctx):\n        return StatePatch()\n"
        "    def before_model(self, view, api):\n        return ViewPatch(changed=False)\n"
        "    def after_proposal(self, action, view, api):\n        return ProposalEffect()\n"
        "    def after_observation(self, event, api):\n        return StatePatch()\n"
        "    def before_finish(self, action, view, api):\n"
        "        return FinishEffect(block=True, block_reason='not allowed yet')\n"
        "component = BlockFinish\n"
    )


def _bank_with_source(tmp_path, source: str):
    """Materialize a one-component bank from source text."""
    from rsi.contracts.artifact import CandidateFile, CandidatePatch
    from rsi.evolution.materializer import materialize_candidate

    empty = load_empty_bank()
    manifest_patch = {
        "id": "testcomp",
        "version": "1",
        "files": ["components/testcomp.py"],
        "entrypoint": "components.testcomp:component",
        "depends_on": [],
        "conflicts_with": [],
        "hook_order": 50,
        "required_capabilities": [],
        "state_schema": {},
        "resource_caps": {"max_source_bytes": 200000},
        "provenance": {"kind": "fixture", "synthetic": True},
        "alignment_hints": {},
        "kind": "decision_hook",
    }
    patch = CandidatePatch(
        parent_bank_digest=empty.bank_digest,
        operation="add",
        hypothesis="test bank",
        files=(CandidateFile(path="components/testcomp.py", content=source),),
        manifest_patch=manifest_patch,
    )
    mat = materialize_candidate(
        patch, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest, out_root=tmp_path
    )
    from rsi.harness.loader import load_bank

    return load_bank(mat.bank_dir)


def _run_teacher(tmp_path, source: str, max_shadow: int = 4):
    task, _ = build_toy_file_tasks(1, 17)[0]
    env = ToyFileWorld()
    env.reset(task, seed=1)
    bank = _bank_with_source(tmp_path, source)
    runner = TeacherForkRunner(
        env_factory=ToyFileWorld,
        policy=ScriptedPolicy(FinisherPlan()),
        bank=bank,
        budgets=BudgetSpec(max_actions=8, max_tool_calls=20),
        max_shadow_events=max_shadow,
    )
    snap = env.snapshot()
    branch = runner.run_branch(
        base_env=env, snapshot_ref=snap, task=task, student_events=(), decision_index=0
    )
    env.close()
    return branch


def test_teacher_veto_feedback(tmp_path):
    """guide §12 test_teacher_veto_feedback: a veto generates a private
    feedback event, consumes budget, and does not loop forever."""
    branch = _run_teacher(tmp_path, _veto_bank_source())
    # the Finish proposal was vetoed → a private veto event must exist
    veto_events = [e for e in branch.events if e.data.get("veto_reason") == "test veto"]
    assert veto_events, "veto must be recorded as a private feedback event"
    assert all(e.visibility == Visibility.TEACHER_PRIVATE for e in veto_events)
    # termination is NOT "finish" (the Finish was vetoed)
    assert branch.terminated_reason != "finish"


def test_teacher_finish_hooks(tmp_path):
    """guide §12 test_teacher_finish_hooks: the teacher cannot bypass
    before_finish; a blocked Finish is not a successful termination."""
    branch = _run_teacher(tmp_path, _finish_block_bank_source())
    # every Finish proposal was blocked → no FINISH event may terminate
    assert branch.terminated_reason != "finish", (
        "a before_finish-blocked Finish must not terminate the branch"
    )
    blocked = [e for e in branch.events if e.data.get("finish_blocked")]
    assert blocked, "block events must be recorded privately"
