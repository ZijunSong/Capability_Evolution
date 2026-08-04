"""Round 7 contract tests: threshold sentinel, sign consistency, decide_dup_operation."""

from __future__ import annotations

import math

import pytest

from harness.capability.dup_operation import DupOperation
from training.scope.decide_dup_operation import COMPARISON_OPERATOR, decide_dup_operation
from training.scope.decision_config import DupDecisionConfig
from training.scope.live_dup_decision_trace import make_trace_from_decision, sha256_text


def test_threshold_positive_inf_all_keep():
    scores = [(0.0, 5.0), (1.0, 3.0), (2.0, 2.0)]
    for sk, ss in scores:
        d = decide_dup_operation(score_keep=sk, score_skip=ss, threshold=float("inf"))
        assert d.predicted_operation == DupOperation.KEEP_EVIDENCE


def test_threshold_negative_inf_all_skip():
    scores = [(5.0, 0.0), (3.0, 1.0), (2.0, 2.0)]
    for sk, ss in scores:
        d = decide_dup_operation(score_keep=sk, score_skip=ss, threshold=float("-inf"))
        assert d.predicted_operation == DupOperation.SKIP_DUPLICATE


def test_threshold_zero_equals_argmax():
    for sk, ss in [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]:
        d = decide_dup_operation(score_keep=sk, score_skip=ss, threshold=0.0)
        expected = DupOperation.SKIP_DUPLICATE if ss >= sk else DupOperation.KEEP_EVIDENCE
        assert d.predicted_operation == expected


def test_increasing_threshold_never_increases_skip():
    scores = [(0.0, 1.0), (1.0, 2.0), (0.5, 0.5)]
    prev_skip = len(scores)
    for th in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        n_skip = sum(
            1
            for sk, ss in scores
            if decide_dup_operation(score_keep=sk, score_skip=ss, threshold=th).predicted_operation
            == DupOperation.SKIP_DUPLICATE
        )
        assert n_skip <= prev_skip
        prev_skip = n_skip


def test_margin_sign_consistency():
    d = decide_dup_operation(score_keep=0.3, score_skip=0.7, threshold=0.0)
    assert d.margin == pytest.approx(0.4)
    assert d.margin == d.score_skip - d.score_keep
    assert d.comparison_operator == COMPARISON_OPERATOR


def test_decision_config_uses_shared_function():
    cfg = DupDecisionConfig(threshold=0.5)
    assert cfg.predict_from_scores(0.0, 1.0) == DupOperation.SKIP_DUPLICATE
    assert cfg.predict_from_scores(1.0, 0.0) == DupOperation.KEEP_EVIDENCE


def test_trace_pre_post_realizer_match():
    trace = make_trace_from_decision(
        query_id="q1",
        turn_index=0,
        decision_index=0,
        decision_state={"query": "test"},
        rendered_prompt="prompt",
        input_ids=[1, 2, 3],
        score_keep=0.1,
        score_skip=0.9,
        threshold=0.0,
        threshold_source="fixed_zero",
        threshold_key="seed42",
        predicted_pre=DupOperation.SKIP_DUPLICATE,
        predicted_post=DupOperation.SKIP_DUPLICATE,
        candidate_evidence_id="cand1",
        shadow_label="SKIP_DUPLICATE",
        shadow_route="ENDORSE",
        actually_curated=False,
        action_payload={"add_ids": []},
        model_id="test",
        checkpoint_path="/path",
        checkpoint_sha256=sha256_text("/path"),
        seed=42,
        backend="vllm_live",
    )
    assert trace.predicted_operation_pre_realizer == trace.predicted_operation_post_realizer
    assert trace.margin == pytest.approx(0.8)
    assert not trace.fallback_used


def test_scorer_uses_candidate_context_in_prompt():
    from unittest.mock import MagicMock

    from training.scope.dup_operation_runtime import DupOperationRuntime, DupOperationRuntimeConfig
    from training.scope.operation_scorer import OperationScoreResult
    from training.scope.prompting import format_operation_prompt

    captured: dict = {}

    class FakeScorer:
        def score(self, text, *, candidate_id=None, curated_document_ids=None):
            captured["candidate_id"] = candidate_id
            captured["prompt"] = format_operation_prompt(
                text, candidate_id=candidate_id, curated_document_ids=curated_document_ids
            )
            return OperationScoreResult(
                scores={
                    DupOperation.KEEP_EVIDENCE.value: 0.0,
                    DupOperation.SKIP_DUPLICATE.value: 1.0,
                },
                predicted=DupOperation.SKIP_DUPLICATE,
                log_probs={},
            )

    rt = DupOperationRuntime(
        None,
        None,
        config=DupOperationRuntimeConfig(decision_config=DupDecisionConfig()),
        vllm_scorer=FakeScorer(),  # type: ignore[arg-type]
    )
    state = MagicMock()
    state.rendered_context = "ctx"
    state.curated_document_ids = ("doc1",)
    state.turn_id = 1
    rt.score_and_predict(state, candidate_id="cand42", curated_document_ids=["doc1"])
    assert captured["candidate_id"] == "cand42"
    assert "cand42" in captured["prompt"]
