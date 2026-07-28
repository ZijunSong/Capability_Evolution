"""Visibility guard tests."""

from __future__ import annotations

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.artifacts.visibility import check_artifact_visibility, mask_artifact_if_invalid
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState


def _state() -> DecisionState:
    return DecisionState(
        episode_id="ep1",
        task_id="t1",
        turn_id=1,
        query="q",
        rendered_context="",
        action_history=(),
        observation_ids=("obs_1",),
        visible_document_ids=("d1",),
        pool_document_ids=("d1",),
        curated_document_ids=("d1",),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=3,
        remaining_search_calls=None,
        token_budget_used=0,
        token_budget_total=100,
        last_action_type=None,
        repeated_query_score=0.0,
        wm_snapshot_hash="h",
    )


def _artifact(**kwargs) -> PrivilegedArtifact:
    student = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={},
    )
    defaults = dict(
        episode_id="ep1",
        turn_id=1,
        module_id="verification",
        mode=GuidanceMode.CORRECT,
        reason_code="PREMATURE_STOP",
        student_action=student,
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.VERIFY_CLAIM,
            arguments={"doc_ids": ["d1"], "claim": "c"},
        ),
        evidence_ids=("obs_1",),
        document_ids=("d1",),
        confidence=0.9,
        metadata={"task_id": "t1"},
    )
    defaults.update(kwargs)
    return PrivilegedArtifact.build(**defaults)


def test_visible_evidence_accepted():
    check = check_artifact_visibility(_state(), _artifact())
    assert check.valid


def test_future_observation_rejected():
    art = _artifact(evidence_ids=("obs_future",))
    check = check_artifact_visibility(_state(), art)
    assert not check.valid
    assert any("evidence_not_visible" in v for v in check.violations)


def test_invisible_document_rejected():
    art = _artifact(document_ids=("secret_doc",))
    check = check_artifact_visibility(_state(), art)
    assert not check.valid


def test_mask_sets_ignore():
    art = _artifact(evidence_ids=("obs_future",))
    masked, check = mask_artifact_if_invalid(_state(), art)
    assert not check.valid
    assert masked.mode == GuidanceMode.IGNORE


def test_recommended_query_hidden_fact():
    art = _artifact(
        recommended_action=CapabilityAction(
            action_type=CapabilityActionType.SEARCH,
            arguments={"query": "leak __HIDDEN__ answer"},
        )
    )
    check = check_artifact_visibility(_state(), art)
    assert not check.valid
