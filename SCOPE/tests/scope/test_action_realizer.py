"""ActionRealizer tests (Case 1)."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.shadow.action_realizer import ActionRealizer
from harness.shadow.evidence_shadow import EvidenceShadow
from training.scope.routing import route_decision
from training.scope.schema import Route

from tests.scope.conftest import make_state


def test_case1_duplicate_correct_skip_curate():
    """Case 1: duplicate curate → CORRECT skip_curate, train_mask=1."""
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    art = EvidenceShadow().analyze(state, student)
    assert art.reason_code == "DUPLICATE_EVIDENCE"

    cand = ActionRealizer().realize(state, art)
    assert cand is not None
    assert cand.action.action_type == CapabilityActionType.CURATE_DOCUMENT

    result = route_decision(state, art, student)
    assert result.route == Route.CORRECT
    assert result.sample.train_mask == 1
    assert result.target_action is not None


def test_action_realizer_deterministic_duplicate():
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
        target="d1",
        recommended_operation="skip_curate",
    )
    cand = ActionRealizer().realize(state, art)
    assert cand is not None
    assert cand.source in {"deterministic", "artifact_recommended"}
