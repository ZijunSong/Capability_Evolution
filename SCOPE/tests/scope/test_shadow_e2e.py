"""End-to-end SCOPE shadow → transition tests."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import DecisionState
from harness.shadow.registry import build_default_registry
from harness.ultra_core import WorkingMemory
from training.opd_v2.pipeline import build_transitions_for_step
from training.train_scope import make_toy_decision_state, run_dry_run
from training.scope_config import load_scope_config, scope_section


def test_selector_verification_before_stop():
    sel = RuleBasedCriticalStateSelector(
        SelectorConfig(verification_enabled=True, evidence_enabled=False)
    )
    state = make_toy_decision_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER, arguments={}
    )
    mods = sel.select(state, action)
    assert "verification" in mods


def test_shadow_does_not_change_wm():
    wm = WorkingMemory("q")
    wm.add_to_pool(["d1"], {"d1": "text"})
    h0 = wm.snapshot_hash()
    registry = build_default_registry(evidence_state=False, verification=True)
    state = make_toy_decision_state(wm_snapshot_hash=h0)
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER, arguments={}
    )
    _ = build_transitions_for_step(state, action, registry=registry)
    # WM untouched
    assert wm.snapshot_hash() == h0
    assert wm.turn_number == 0


def test_episode_can_generate_correct_transition():
    registry = build_default_registry(evidence_state=False, verification=True)
    state = make_toy_decision_state(curated_document_ids=())
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER, arguments={}
    )
    trs = build_transitions_for_step(state, action, registry=registry)
    assert any(t.mode == GuidanceMode.CORRECT for t in trs)


def test_episode_can_generate_endorse_on_verify():
    registry = build_default_registry(evidence_state=False, verification=True)
    state = make_toy_decision_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.VERIFY_CLAIM,
        arguments={"doc_ids": ["doc_a"], "claim": "founder"},
    )
    trs = build_transitions_for_step(state, action, registry=registry)
    assert any(t.mode in {GuidanceMode.ENDORSE, GuidanceMode.CORRECT} for t in trs)


def test_dry_run_train_scope(tmp_path):
    cfg = load_scope_config("configs/scope/verification_only.yaml")
    scope = scope_section(cfg)
    metrics = run_dry_run(scope, tmp_path)
    assert "opd_total" in metrics
    assert (tmp_path / "toy_transitions.jsonl").exists()


def test_modules_can_be_disabled():
    registry = build_default_registry(
        evidence_state=False, verification=False, budget_control=False
    )
    assert registry.ids() == []
