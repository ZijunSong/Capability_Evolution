"""VerifiedDecisionRouting tests (Cases 2, 5, 6, 7)."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.shadow.verification_shadow import VerificationShadow
from training.scope.routing import route_decision
from training.scope.schema import Route

from tests.scope.conftest import make_state, verified_stop_state


def test_case2_endorse_correct_rejection():
    """Case 2: student correctly rejects duplicate → ENDORSE."""
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    art = EvidenceShadow().analyze(state, student)
    assert art.mode == GuidanceMode.ENDORSE
    result = route_decision(state, art, student)
    assert result.route == Route.ENDORSE
    assert result.target_action is not None
    assert result.target_action.canonical_key() == student.canonical_key()


def test_case5_premature_stop_correct():
    """Case 5: premature stop → continue_search/verify → CORRECT."""
    from harness.capability.state import VerificationRecordState

    state = make_state(
        verification_records=(
            VerificationRecordState(
                turn_id=1,
                claim="x",
                document_ids=("d1",),
                judgments={"d1": False},
            ),
        ),
        curated_document_ids=("d1",),
    )
    student = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"answer": "maybe"},
    )
    art = VerificationShadow().analyze(state, student)
    assert art.capability_id == "premature_stop"
    result = route_decision(state, art, student)
    assert result.route == Route.CORRECT
    assert result.sample.train_mask == 1


def test_case6_normal_stop_endorse():
    """Case 6: adequate evidence stop → ENDORSE."""
    state = verified_stop_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"answer": "Y invented X"},
    )
    art = VerificationShadow().analyze(state, student)
    assert art.mode == GuidanceMode.ENDORSE
    result = route_decision(state, art, student)
    assert result.route == Route.ENDORSE
    assert result.sample.train_mask == 1


def test_case7_illegal_action_ignore():
    """Case 7: shadow suggests forbidden op → IGNORE, executable=false."""
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    art = PrivilegedArtifact.build(
        episode_id="ep1",
        turn_id=3,
        module_id="evidence_state",
        mode=GuidanceMode.CORRECT,
        reason_code="DUPLICATE_EVIDENCE",
        student_action=student,
        evidence_ids=("obs_1",),
        document_ids=("d1",),
        capability_id="duplicate_evidence",
        recommended_operation="stop_and_answer",
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.STOP_AND_ANSWER,
            arguments={"answer": "hidden"},
        ),
    )
    result = route_decision(state, art, student)
    assert result.route == Route.IGNORE
    assert not result.gates.executable or result.sample.train_mask == 0
