"""Unit tests for expanded Evidence corrections and Verification stop/answer capture."""

from __future__ import annotations

from harness.artifacts.reason_codes import EVIDENCE_REASON_CODES
from harness.artifacts.schema import GuidanceMode
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.adapters import parse_action_from_tools
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import ClaimState, VerificationRecordState
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.shadow.verification_shadow import VerificationShadow
from training.train_scope import make_toy_decision_state


def test_evidence_reason_codes_include_required_set():
    required = {
        "DUPLICATE_EVIDENCE",
        "IRRELEVANT_EVIDENCE",
        "WEAK_SUPPORT",
        "MISSING_DIRECT_SUPPORT",
        "CONFLICTING_EVIDENCE",
        "WRONG_CLAIM_BINDING",
    }
    assert required <= set(EVIDENCE_REASON_CODES)


def test_evidence_duplicate_correction():
    shadow = EvidenceShadow()
    state = make_toy_decision_state(curated_document_ids=("doc_a",))
    action = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["doc_a"], "remove_ids": []},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code == "DUPLICATE_EVIDENCE"


def test_evidence_missing_direct_support():
    shadow = EvidenceShadow()
    state = make_toy_decision_state(
        evidence_claims=(
            ClaimState(
                claim_id="c1",
                text="Acme founded by X",
                status="unverified",
                supporting_document_ids=(),
            ),
        )
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.UPDATE_EVIDENCE,
        arguments={"status": "supported"},
        target_claim_id="c1",
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code in {
        "MISSING_DIRECT_SUPPORT",
        "INVALID_STATUS_TRANSITION",
        "WEAK_SUPPORT",
    }


def test_evidence_conflicting():
    shadow = EvidenceShadow()
    state = make_toy_decision_state(
        verification_records=(
            VerificationRecordState(
                turn_id=1,
                claim="founder",
                document_ids=("doc_a", "doc_b"),
                judgments={"doc_a": True, "doc_b": False},
            ),
        ),
        evidence_claims=(
            ClaimState(
                claim_id="c1",
                text="founder",
                status="conflict",
                supporting_document_ids=("doc_a", "doc_b"),
            ),
        ),
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["doc_b"], "remove_ids": []},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code == "CONFLICTING_EVIDENCE"


def test_evidence_wrong_claim_binding():
    shadow = EvidenceShadow()
    state = make_toy_decision_state(
        pool_document_ids=("doc_a",),
        visible_document_ids=("doc_a",),
        curated_document_ids=("doc_a",),
        evidence_claims=(
            ClaimState(
                claim_id="c1",
                text="founder",
                status="linked",
                supporting_document_ids=("doc_missing",),
            ),
        ),
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.UPDATE_EVIDENCE,
        arguments={"status": "linked", "add_ids": ["doc_missing"]},
        target_claim_id="c1",
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code == "WRONG_CLAIM_BINDING"


def test_selector_captures_stop_answer_verify():
    sel = RuleBasedCriticalStateSelector(
        SelectorConfig(verification_enabled=True, evidence_enabled=False)
    )
    state = make_toy_decision_state()
    for atype in (
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.VERIFY_CLAIM,
    ):
        action = CapabilityAction(action_type=atype, arguments={"doc_ids": ["doc_a"], "claim": "x"})
        mods = sel.select(state, action)
        assert "verification" in mods, atype


def test_user_text_maps_to_answer():
    cap = parse_action_from_tools(["user_text"], [{"text": "final answer is 42"}])
    assert cap is not None
    assert cap.action_type == CapabilityActionType.ANSWER


def test_verification_corrects_insufficient_evidence_on_stop():
    shadow = VerificationShadow()
    state = make_toy_decision_state(
        curated_document_ids=(),
        verification_records=(),
        evidence_claims=(),
    )
    for atype in (
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
    ):
        action = CapabilityAction(
            action_type=atype,
            arguments={"reasoning": "done", "text": "42"},
        )
        art = shadow.analyze(state, action)
        assert art.mode == GuidanceMode.CORRECT, atype
        assert art.reason_code in {"MISSING_DIRECT_EVIDENCE", "PREMATURE_STOP"}, atype


def test_verification_corrects_stop_without_verify_records():
    shadow = VerificationShadow()
    # Has curated docs but never verified — premature stop
    state = make_toy_decision_state(
        curated_document_ids=("doc_a",),
        verification_records=(),
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": "enough"},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code == "PREMATURE_STOP"
    assert art.recommended_action is not None
    assert art.recommended_action.action_type == CapabilityActionType.VERIFY_CLAIM


def test_verification_endorses_careful_verify():
    shadow = VerificationShadow()
    state = make_toy_decision_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.VERIFY_CLAIM,
        arguments={"doc_ids": ["doc_a"], "claim": "founder"},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.ENDORSE
    assert art.reason_code == "VERIFICATION_SUPPORTED"


def test_verification_endorses_valid_stop_with_positive_verify():
    shadow = VerificationShadow()
    state = make_toy_decision_state(
        curated_document_ids=("doc_a",),
        visible_document_ids=("doc_a",),
        pool_document_ids=("doc_a",),
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
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": "sufficient"},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.ENDORSE
    assert art.reason_code == "VERIFICATION_SUPPORTED"


def test_verification_corrects_premature_stop_without_verify():
    shadow = VerificationShadow()
    state = make_toy_decision_state(
        curated_document_ids=("doc_a",),
        verification_records=(),
    )
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": "done"},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.CORRECT
    assert art.reason_code == "PREMATURE_STOP"


def test_verification_ignores_plain_curate():
    shadow = VerificationShadow()
    state = make_toy_decision_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["doc_b"], "remove_ids": []},
    )
    art = shadow.analyze(state, action)
    assert art.mode == GuidanceMode.IGNORE
