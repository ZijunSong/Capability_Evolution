"""Shadow purity gate tests (Case 4)."""

from __future__ import annotations

from harness.artifacts.gates import run_information_safe_gates, shadow_purity_gate
from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from training.scope.routing import route_decision
from training.scope.schema import Route

from tests.scope.conftest import make_state


def test_shadow_purity_gate_equal_fingerprints():
    fp = {"wm_hash": "abc", "n_curated": 1}
    gate = shadow_purity_gate(fp, fp)
    assert gate.passed


def test_case4_shadow_mutation_forces_ignore():
    """Case 4: WM changed after shadow → purity fail → IGNORE."""
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
        recommended_operation="skip_curate",
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": []},
        ),
    )
    before = {"wm_hash": "before", "n_curated": 1}
    after = {"wm_hash": "after", "n_curated": 2}
    report = run_information_safe_gates(
        state, art, fingerprint_before=before, fingerprint_after=after
    )
    assert not report.purity_ok
    result = route_decision(
        state,
        art,
        student,
        fingerprint_before=before,
        fingerprint_after=after,
    )
    assert result.route == Route.IGNORE
    assert result.sample.train_mask == 0
    assert result.gates.audit_error == "SHADOW_MUTATED_ENV"
