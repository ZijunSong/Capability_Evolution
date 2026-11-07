"""PR-4 effect telemetry + EvidenceStore (audit doc §5.2, §7.2, §9.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.observation import Observation, StepResult
from rsi.contracts.task import PrivateTaskRef, PublicTask
from rsi.evolution.evidence_store import EvidenceStore
from rsi.harness.composition import ComponentEffectRecord, HarnessComposition
from rsi.harness.component import ViewPatch
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop
from rsi.providers.scripted import ScriptedPolicy
from rsi.harness.loader import load_empty_bank


class _NoOpComponent:
    """A hook that is CALLED but never changes anything."""

    def initialize(self, ctx):
        pass

    def before_model(self, view, api):
        return ViewPatch(changed=False)

    def after_proposal(self, action, view, api):
        pass

    def after_observation(self, event, api):
        pass

    def before_finish(self, action, view, api):
        pass


class _ActiveComponent:
    """A hook that actually rewrites the proposal on the first decision."""

    def __init__(self):
        self.calls = 0

    def initialize(self, ctx):
        pass

    def before_model(self, view, api):
        return ViewPatch(append_instructions="be concise", changed=True)

    def after_proposal(self, action, view, api):
        self.calls += 1
        if self.calls == 1 and isinstance(action, ToolCall):
            from rsi.harness.component import ProposalEffect

            return ProposalEffect(
                action=ToolCall(name=action.name, arguments=dict(action.arguments)),
                veto=False,
                notes={"rewritten": True},
            )
        return None

    def after_observation(self, event, api):
        pass

    def before_finish(self, action, view, api):
        pass


class _ToyEnv:
    def tool_catalog(self):
        return ()

    def reset(self, task, seed):
        return Observation(content="ready")

    def step(self, action):
        if isinstance(action, Finish):
            return StepResult(
                observation=Observation(content="done"), done=True,
                snapshot=self.snapshot(), cost={}
            )
        return StepResult(
            observation=Observation(content="ok"), done=False,
            snapshot=self.snapshot(), cost={}
        )

    def snapshot(self):
        from rsi.contracts.observation import SnapshotRef

        return SnapshotRef(ref_id="snap0", digest="d0", kind="checkpoint")

    def state_digest(self):
        return "d"

    def close(self):
        pass


_TASK = PublicTask(
    task_id="toy:t1", domain="toy", instruction="do it", group_id="toy:g", metadata={}
)


def _policy_finish_now():
    def plan(view_text: str, step: int):
        if step == 0:
            return ToolCall(name="noop", arguments={})
        return Finish(answer=None)

    return ScriptedPolicy(plan)


def _run_with_bank(tmp_path, components: list):
    """Build a real bank dir and run one scripted episode through the loop."""
    from rsi.harness.loader import load_bank

    manifest = {
        "bank_id": "test-bank",
        "schema_version": "rsi_bank_manifest_v1",
        "hooks": [],
        "tools": [],
        "skills": [],
        "components": [
            {
                "id": type(c).__name__.lower(),
                "kind": "other",
                "entrypoint": f"components.{type(c).__name__.lower()}:{type(c).__name__}",
                "files": [f"components/{type(c).__name__.lower()}.py"],
                "hook_order": 100,
                "version": "1",
                "state_schema": {},
                "alignment_hints": {},
                "provenance": {},
                "required_capabilities": [],
                "resource_caps": {},
                "conflicts_with": [],
                "depends_on": [],
            }
            for c in components
        ],
        "extra": {},
        "parent_digest": None,
    }
    import json as _json

    src = {
        _NoOpComponent: (
            "from rsi.harness.component import ViewPatch\n\n"
            "class _NoOpComponent:\n"
            "    def initialize(self, ctx): pass\n"
            "    def before_model(self, view, api): return ViewPatch(changed=False)\n"
            "    def after_proposal(self, action, view, api): pass\n"
            "    def after_observation(self, event, api): pass\n"
            "    def before_finish(self, action, view, api): pass\n"
        ),
        _ActiveComponent: (
            "from rsi.harness.component import ViewPatch, ProposalEffect\n"
            "from rsi.contracts.action import ToolCall\n\n"
            "class _ActiveComponent:\n"
            "    def __init__(self): self.calls = 0\n"
            "    def initialize(self, ctx): pass\n"
            "    def before_model(self, view, api):\n"
            "        return ViewPatch(append_instructions='be concise', changed=True)\n"
            "    def after_proposal(self, action, view, api):\n"
            "        self.calls += 1\n"
            "        if self.calls == 1 and isinstance(action, ToolCall):\n"
            "            return ProposalEffect(action=ToolCall(name=action.name, "
            "arguments=dict(action.arguments)), veto=False, notes={'rewritten': True})\n"
            "        return None\n"
            "    def after_observation(self, event, api): pass\n"
            "    def before_finish(self, action, view, api): pass\n"
        ),
    }
    bank_dir = tmp_path / f"bank_{len(components)}"
    comps = []
    for c in components:
        name = type(c).__name__.lower()
        d = bank_dir / "components"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.py").write_text(src[type(c)], "utf-8")
        comps.append(
            {
                "id": name,
                "kind": "other",
                "entrypoint": f"components.{name}:{type(c).__name__}",
                "files": [f"components/{name}.py"],
                "hook_order": 100,
                "version": "1",
                "state_schema": {},
                "alignment_hints": {},
                "provenance": {},
                "required_capabilities": [],
                "resource_caps": {},
                "conflicts_with": [],
                "depends_on": [],
            }
        )
    manifest["components"] = comps
    (bank_dir / "manifest.json").write_text(_json.dumps(manifest), "utf-8")
    bank = load_bank(bank_dir)
    loop = MinimalLoop(
        env=_ToyEnv(),
        policy=_policy_finish_now(),
        bank=bank,
        budgets=BudgetSpec(max_actions=3),
    )
    return loop.run_episode(_TASK, seed=1)


def test_noop_hook_invoked_not_changed(tmp_path):
    """Audit §14 effect telemetry: no-op hook -> invoked>0, changed==0."""
    result = _run_with_bank(tmp_path, [_NoOpComponent()])
    effects = result.component_effects
    assert effects, "effect records missing"
    noop = effects.get("_noopcomponent") or next(iter(effects.values()))
    assert noop["invoked"] > 0
    assert noop["changed_view"] == 0
    assert noop["changed_action"] == 0


def test_active_hook_changed_view_and_action(tmp_path):
    result = _run_with_bank(tmp_path, [_ActiveComponent()])
    effects = result.component_effects
    active = effects.get("_activecomponent") or next(iter(effects.values()))
    assert active["changed_view"] > 0
    assert active["changed_action"] > 0


def test_evidence_store_role_gate():
    store = EvidenceStore()
    with pytest.raises(Exception):
        store.add_panel_result(
            role="final_test", per_task={"t1": {"value": 0.0, "terminated_reason": "budget_wall"}}
        )
    # readable role works
    n = store.add_panel_result(
        role="harness_select",
        per_task={"t1": {"value": 0.0, "terminated_reason": "budget_wall", "n_decisions": 3, "wall_ms": 12.0}},
    )
    assert n == 1
    fragments = store.failure_fragments(role="harness_select")
    assert fragments and "budget_wall" in fragments[0]


def test_packet_fragments_carried_into_evolve(tmp_path):
    """search._fragments_from_per_task fills the packet's fragment fields."""
    from rsi.evolution.search import EvolutionSearch

    per_task = {
        "t_bad": {
            "value": 0.0,
            "success": False,
            "terminated_reason": "max_actions",
            "n_decisions": 24,
            "wall_ms": 1234.0,
            "metric_name": "appworld_strict_success",
        },
        "t_good": {
            "value": 1.0,
            "success": True,
            "terminated_reason": "finish",
            "n_decisions": 5,
            "wall_ms": 100.0,
        },
    }
    fails = EvolutionSearch._fragments_from_per_task(per_task, failure=True)
    succ = EvolutionSearch._fragments_from_per_task(per_task, failure=False)
    assert len(fails) == 1 and fails[0].task_id == "t_bad"
    assert "n_decisions=24" in fails[0].events_text
    assert len(succ) == 1 and succ[0].task_id == "t_good"
