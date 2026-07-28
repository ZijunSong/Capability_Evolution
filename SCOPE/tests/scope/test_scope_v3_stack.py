"""Tests for SCOPE v3 core stack: capability / state / artifact / gates / routing / SDI."""

from __future__ import annotations

import torch

from harness.artifacts.gates import run_information_safe_gates
from harness.artifacts.provenance import ProvenanceKind, assert_info_subset
from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import (
    CapabilityId,
    is_round1_trainable,
    parse_capability_id,
)
from harness.capability.state import DecisionState, SCHEMA_VERSION
from harness.shadow.action_realizer import ActionRealizer
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.shadow.verification_shadow import VerificationShadow
from training.scope.losses import action_span_labels, compute_sdi_loss
from training.scope.routing import route_decision
from training.scope.schema import Route


def _state(**kwargs) -> DecisionState:
    base = dict(
        episode_id="ep1",
        task_id="t1",
        turn_id=3,
        query="Who invented X?",
        rendered_context="doc d1 mentions X inventor",
        action_history=(),
        observation_ids=("obs_1",),
        visible_document_ids=("d1", "d2"),
        pool_document_ids=("d1", "d2"),
        curated_document_ids=("d1",),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=5,
        remaining_search_calls=None,
        token_budget_used=10,
        token_budget_total=100,
        last_action_type="curate_document",
        repeated_query_score=0.0,
        wm_snapshot_hash="h",
    )
    base.update(kwargs)
    return DecisionState(**base)


def test_capability_round1_filter():
    assert is_round1_trainable(CapabilityId.DUPLICATE_EVIDENCE)
    assert is_round1_trainable("premature_stop")
    assert not is_round1_trainable(CapabilityId.IRRELEVANT_EVIDENCE)
    assert parse_capability_id("DUPLICATE_EVIDENCE") == CapabilityId.DUPLICATE_EVIDENCE


def test_decision_state_v2_fields_and_safety():
    s = _state(goal="Who invented X?", last_action_arguments={"add_ids": ["d2"]})
    assert s.schema_version == SCHEMA_VERSION
    assert s.observed_ids == ("obs_1",)
    d = s.to_dict()
    assert "observed_ids" in d
    assert "last_action_arguments" in d
    ok, bad = s.check_info_safety()
    assert ok
    assert not bad
    assert s.field_provenance()["remaining_turns"] == ProvenanceKind.RUNTIME.value
    s2 = DecisionState.from_dict(d)
    assert s2.core_state_hash() == s.core_state_hash()


def test_assert_info_subset_rejects_forbidden():
    ok, bad = assert_info_subset(["query", "gold_answer"])
    assert not ok
    assert "gold_answer" in bad


def test_artifact_v3_duplicate():
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
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": [], "remove_ids": ["d1"]},
        ),
        evidence_ids=("obs_1",),
        document_ids=("d1",),
        capability_id="duplicate_evidence",
        recommended_operation="skip_curate",
        teacher_source="EvidenceShadow",
    )
    assert art.schema_version.startswith("scope.artifact")
    assert art.capability_id == "duplicate_evidence"
    assert art.resolved_capability() == CapabilityId.DUPLICATE_EVIDENCE


def test_gates_visibility_and_responsibility():
    state = _state()
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
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": [], "remove_ids": []},
        ),
        evidence_ids=("obs_1",),
        document_ids=("d1",),
        capability_id="duplicate_evidence",
        recommended_operation="skip_curate",
    )
    report = run_information_safe_gates(state, art, candidate_action=art.recommended_action)
    assert report.visible
    assert report.module_valid
    assert report.all_passed

    bad = PrivilegedArtifact.build(
        episode_id="ep1",
        turn_id=3,
        module_id="evidence_state",
        mode=GuidanceMode.CORRECT,
        reason_code="DUPLICATE_EVIDENCE",
        student_action=student,
        evidence_ids=("obs_future",),
        capability_id="duplicate_evidence",
        recommended_operation="skip_curate",
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": []},
        ),
    )
    report2 = run_information_safe_gates(state, bad)
    assert not report2.visible


def test_action_realizer_duplicate_deterministic():
    state = _state()
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
    assert cand.action.action_type == CapabilityActionType.CURATE_DOCUMENT
    assert cand.source in {"deterministic", "artifact_recommended"}


def test_routing_correct_duplicate():
    state = _state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    shadow = EvidenceShadow()
    art = shadow.analyze(state, student)
    assert art.reason_code == "DUPLICATE_EVIDENCE"
    assert art.capability_id == "duplicate_evidence"
    result = route_decision(state, art, student)
    assert result.route == Route.CORRECT
    assert result.sample.train_mask == 1
    assert result.sample.target_action is not None
    assert result.sample.schema_version == "scope.supervision.v3"


def test_routing_filters_irrelevant():
    state = _state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    art = PrivilegedArtifact.build(
        episode_id="ep1",
        turn_id=3,
        module_id="evidence_state",
        mode=GuidanceMode.CORRECT,
        reason_code="IRRELEVANT_EVIDENCE",
        student_action=student,
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.CURATE_DOCUMENT,
            arguments={"add_ids": [], "remove_ids": ["d2"]},
        ),
        evidence_ids=("obs_1",),
        document_ids=("d2",),
        capability_id="irrelevant_evidence",
    )
    result = route_decision(state, art, student)
    assert result.route == Route.IGNORE
    assert result.sample.train_mask == 0
    assert result.sample.audit_error == "CAPABILITY_DISABLED_ROUND1"


def test_premature_stop_capability_on_stop():
    from harness.capability.state import VerificationRecordState

    state = _state(
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
    assert art.mode == GuidanceMode.CORRECT


def test_sdi_loss_action_span():
    bsz, seq, vocab = 2, 8, 20
    logits = torch.randn(bsz, seq, vocab)
    input_ids = torch.randint(0, vocab, (bsz, seq))
    attn = torch.ones(bsz, seq, dtype=torch.long)
    labels = action_span_labels(input_ids, attn, [4, 5], [7, 7])
    # Prompt tokens ignored
    assert (labels[:, :4] == -100).all()
    out = compute_sdi_loss(
        logits,
        labels,
        sample_weights=torch.tensor([1.0, 1.0]),
    )
    assert out.n_active == 2
    assert torch.isfinite(out.loss)
