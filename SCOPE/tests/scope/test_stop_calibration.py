"""Tests for bilateral Stop Calibration selector + verification shadow."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import ClaimState, VerificationRecordState
from harness.capability.stop_calibration import (
    compute_stop_calibration_signals,
    should_trigger_stop_calibration,
)
from harness.shadow.verification_shadow import VerificationShadow
from training.train_scope import make_toy_decision_state


def _sufficient_state():
    return make_toy_decision_state(
        curated_document_ids=("doc_a", "doc_b"),
        visible_document_ids=("doc_a", "doc_b"),
        pool_document_ids=("doc_a", "doc_b"),
        evidence_claims=(
            ClaimState(
                claim_id="c1",
                text="founder",
                status="supported",
                supporting_document_ids=("doc_a",),
            ),
        ),
        verification_records=(
            VerificationRecordState(
                turn_id=1,
                claim="founder",
                document_ids=("doc_a",),
                judgments={"doc_a": True},
            ),
        ),
        action_history=(
            __import__("harness.capability.state", fromlist=["ActionRecord"]).ActionRecord(
                turn_id=1, action_type="search", arguments={"query": "founder"}
            ),
            __import__("harness.capability.state", fromlist=["ActionRecord"]).ActionRecord(
                turn_id=2, action_type="search", arguments={"query": "founder again"}
            ),
        ),
        remaining_turns=4,
    )


def test_stop_calibration_signals_sufficient():
    state = _sufficient_state()
    signals = compute_stop_calibration_signals(state)
    assert signals.sufficient_for_stop is True
    assert signals.coverage_score >= 0.75


def test_selector_triggers_on_high_coverage_continue():
    sel = RuleBasedCriticalStateSelector(
        SelectorConfig(
            verification_enabled=True,
            evidence_enabled=False,
            stop_calibration=True,
        )
    )
    state = _sufficient_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.SEARCH,
        arguments={"query": "more"},
    )
    assert should_trigger_stop_calibration(state, action)[0] is True
    mods = sel.select(state, action)
    assert "verification" in mods
    triggers = [t.trigger for t in sel.last_triggers]
    assert any("high_coverage" in t or "low_budget" in t for t in triggers)


def test_verification_corrects_continue_to_stop_when_sufficient():
    shadow = VerificationShadow()
    state = _sufficient_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.SEARCH,
        arguments={"query": "extra"},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.recommended_action is not None
    assert art.recommended_action.action_type == CapabilityActionType.STOP_AND_ANSWER
    assert art.resolved_capability().value == "premature_stop"


def test_verification_still_corrects_premature_stop():
    shadow = VerificationShadow()
    state = make_toy_decision_state(
        curated_document_ids=(),
        verification_records=(),
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": "done"},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code in {"MISSING_DIRECT_EVIDENCE", "PREMATURE_STOP"}
