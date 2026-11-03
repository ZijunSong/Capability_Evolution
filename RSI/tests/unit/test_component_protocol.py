"""Hook-signature validation and component-runtime-error tolerance.

Regression for the 2026-10-23 r2 failure: a real model-generated candidate
(``CompletionSynthesizer``) defined ``before_model(self, view)`` and crashed
panel workers with TypeError, aborting the whole evolve run.  Signature
mismatches are now ``candidate_invalid`` at load time; runtime failures
disable + record the component without killing the episode or the pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.contracts.artifact import CandidateFile, CandidatePatch
from rsi.contracts.errors import CandidateInvalid
from rsi.evolution.materializer import materialize_candidate
from rsi.evolution.validator import validate_candidate_files, validate_materialized
from rsi.harness.loader import load_bank, load_empty_bank
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop
from rsi.environments.toy import ToyFileWorld, build_toy_file_tasks
from rsi.providers.scripted import ScriptedPolicy

BAD_SIG_SOURCE = '''
from rsi.harness.component import ViewPatch


class CompletionSynthesizer:
    def initialize(self, ctx):
        pass

    def before_model(self, view):  # missing api arg -> protocol invalid
        return ViewPatch(changed=False)

    def after_proposal(self, action, view, api):
        from rsi.harness.component import ProposalEffect
        return ProposalEffect(action=None)

    def after_observation(self, event, api):
        from rsi.harness.component import StatePatch
        return StatePatch()

    def before_finish(self, action, view, api):
        from rsi.harness.component import FinishEffect
        return FinishEffect()
'''

RUNTIME_RAISE_SOURCE = '''
from rsi.harness.component import FinishEffect, ProposalEffect, PublicView, RuntimeAPI, StatePatch, ViewPatch


class Exploder:
    def initialize(self, ctx):
        pass

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        raise RuntimeError("boom in hook")

    def after_proposal(self, action, view, api) -> ProposalEffect:
        return ProposalEffect(action=None)

    def after_observation(self, event, api) -> StatePatch:
        return StatePatch()

    def before_finish(self, action, view, api) -> FinishEffect:
        return FinishEffect()
'''


def _patch(source: str, comp_id: str, parent_digest: str) -> CandidatePatch:
    return CandidatePatch(
        parent_bank_digest=parent_digest,
        operation="add",
        hypothesis="test",
        files=(CandidateFile(path=f"components/{comp_id}.py", content=source),),
        manifest_patch={
            "id": comp_id,
            "version": "1",
            "files": [f"components/{comp_id}.py"],
            "entrypoint": f"components.{comp_id}:{comp_id.title().replace('_', '')}",
            "depends_on": [],
            "conflicts_with": [],
            "hook_order": 50,
            "required_capabilities": [],
            "state_schema": {},
            "resource_caps": {},
            "provenance": {},
            "alignment_hints": {},
            "kind": "decision_hook",
        },
    )


def test_bad_hook_signature_rejected_at_validation():
    empty = load_empty_bank()
    patch = _patch(BAD_SIG_SOURCE, "completion_synthesizer", empty.bank_digest)
    report = validate_candidate_files(
        files={"components/completion_synthesizer.py": BAD_SIG_SOURCE},
        manifest_patch=patch.manifest_patch,
    )
    assert not report.ok
    assert any("before_model" in p and "positional args" in p for p in report.problems), report.problems


def test_bad_hook_signature_rejected_at_load(tmp_path):
    empty = load_empty_bank()
    patch = _patch(BAD_SIG_SOURCE, "completion_synthesizer", empty.bank_digest)
    # materialize_candidate verifies by loading, so the loader bind-check fires
    # there (and again if a stale dir is loaded directly).
    with pytest.raises(CandidateInvalid, match="before_model"):
        materialize_candidate(
            patch, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest, out_root=tmp_path
        )


def test_runtime_component_error_disables_not_crashes():
    """A component that raises in a hook is disabled + recorded; the episode
    finishes and the panel marks protocol failure instead of aborting."""
    empty = load_empty_bank()
    patch = _patch(RUNTIME_RAISE_SOURCE, "exploder", empty.bank_digest)
    mat = materialize_candidate(
        patch, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest,
        out_root=Path("runs/tmp_proto_test"),
    )
    report = validate_materialized(mat.bank_dir)
    assert report.ok, report.problems
    bank = load_bank(mat.bank_dir)

    task, _ = build_toy_file_tasks(1, 17)[0]
    env = ToyFileWorld()
    loop = MinimalLoop(
        env=env,
        policy=ScriptedPolicy(lambda view, step: __import__("rsi.contracts.action", fromlist=["Finish"]).Finish(answer=None)),
        bank=bank,
        budgets=BudgetSpec(max_actions=4),
    )
    res = loop.run_episode(task, seed=1)
    # episode completed despite the exploding component
    assert res.trace.terminated_reason in ("finish", "max_actions")
    # component was disabled and the error recorded (not silently active)
    assert "exploder" in res.component_errors
    assert "boom in hook" in res.component_errors["exploder"]


def test_entrypoint_class_mismatch_rejected():
    """manifest entrypoint naming a class the source does not define is caught
    at validation (was: only failed at load with a vague error)."""
    src = (
        "from rsi.harness.component import FinishEffect, ProposalEffect, StatePatch, ViewPatch\n\n\n"
        "class TerminationPolicyHook:\n"
        "    def initialize(self, ctx):\n        pass\n\n"
        "    def before_model(self, view, api):\n        return ViewPatch(changed=False)\n\n"
        "    def after_proposal(self, action, view, api):\n        return ProposalEffect(action=None)\n\n"
        "    def after_observation(self, event, api):\n        return StatePatch()\n\n"
        "    def before_finish(self, action, view, api):\n        return FinishEffect()\n"
    )
    report = validate_candidate_files(
        files={"components/termination_policy.py": src},
        manifest_patch={
            "id": "termination_policy",
            "version": "1",
            "files": ["components/termination_policy.py"],
            "entrypoint": "components.termination_policy:TerminationPolicy",  # wrong: class is TerminationPolicyHook
            "depends_on": [], "conflicts_with": [], "hook_order": 50,
            "required_capabilities": [], "state_schema": {}, "resource_caps": {},
            "provenance": {}, "alignment_hints": {}, "kind": "decision_hook",
        },
    )
    assert not report.ok
    assert any("TerminationPolicyHook" in p for p in report.problems), report.problems


def test_repair_retries_carry_feedback(tmp_path):
    """Failed attempts must feed the rejection reason back to the proposer
    (was: blind retries repeating the same mistake)."""
    from rsi.evolution.search import EvolutionBudgets, EvolutionSearch
    from rsi.evolution.archive import Archive

    empty = load_empty_bank()
    good = _patch(RUNTIME_RAISE_SOURCE, "exploder", empty.bank_digest)  # valid signatures
    bad = _patch(BAD_SIG_SOURCE, "completion_synthesizer", empty.bank_digest)

    class RecordingProposer:
        mode = "fixture"

        def __init__(self):
            self.notes: list[tuple[str, ...]] = []
            self.calls = 0

        def propose(self, packet, *, retry_notes=()):
            self.notes.append(tuple(retry_notes))
            self.calls += 1
            return bad if self.calls == 1 else good

    proposer = RecordingProposer()

    def panel_runner(bank):
        # PR-5 semantics: the parent (empty bank) is measured too, and a
        # candidate only becomes champion when it EXCEEDS the parent's macro.
        # The good patch outperforms the parent 1.0 vs 0.5 here.
        if bank.bank_digest == empty.bank_digest:
            value = 0.5
        else:
            value = 1.0
        return {
            "scores": {"toy_file": {"t1": value}},
            "per_task": {"t1": {"value": value, "terminated_reason": "finish", "trigger_stats": {}}},
            "cost": {},
            "protocol_ok": True,
            "budget_ok": True,
        }

    from rsi.evolution.packet import build_packet

    search = EvolutionSearch(
        proposer=proposer,
        archive=Archive(tmp_path / "archive"),
        panel_runner=panel_runner,
        packet_builder=build_packet,
        budgets=EvolutionBudgets(max_generations=1, candidates_per_generation=1,
                                 max_candidate_evaluations=1, plateau_patience=1,
                                 max_candidate_repairs=2),
        candidates_root=tmp_path / "candidates",
        known_task_ids=(),
    )
    result = search.run()
    # first attempt (bad signature) rejected; second attempt carries the reason
    assert proposer.calls >= 2
    assert proposer.notes[0] == ()
    assert proposer.notes[1], "retry must carry the rejection reason"
    assert any("before_model" in n or "positional" in n for n in proposer.notes[1])
    assert result.champion is not None and result.champion.candidate_id
