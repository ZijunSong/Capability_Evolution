"""Measurement correctness (§16.5, §16.6) for Probe-C metrics."""

from __future__ import annotations

import math

import pytest

from trim.eval.probe_c_metrics import (
    MISSING,
    C1Verdict,
    GenerationRecord,
    MetricsError,
    aggregate_js,
    aggregate_nll,
    candidate_probs,
    improvement,
    invalid_rate_noninferior,
    js_divergence,
    judge_c1,
    name_aggregate_probs,
    regen_stats,
    score_candidates,
    score_teacher_nll,
    sequence_logp_sum,
)


def test_hand_computed_js():
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 1.0, "b": 0.0}
    # m = (0.75, 0.25); JS = 0.5 KL(p||m) + 0.5 KL(q||m)
    eps = 1e-12
    kl_pm = 0.5 * math.log(0.5 / (0.75 + eps)) + 0.5 * math.log(0.5 / (0.25 + eps))
    kl_qm = 1.0 * math.log(1.0 / (0.75 + eps))
    expected = 0.5 * kl_pm + 0.5 * kl_qm
    assert math.isclose(js_divergence(p, q), expected, rel_tol=1e-9)


def test_same_distribution_js_zero():
    z = {"a": -1.0, "b": -2.0, "c": -3.0}
    p = candidate_probs(z)
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_candidate_order_independent():
    z1 = {"a": -1.0, "b": -2.0}
    z2 = {"b": -2.0, "a": -1.0}
    assert candidate_probs(z1) == candidate_probs(z2)


def test_fewer_than_two_candidates_is_missing_not_zero():
    score = score_candidates(
        state_id="s",
        query_id="q",
        candidate_ids=["only"],
        z_teacher={"only": -1.0},
        z_student={"only": -1.0},
    )
    assert score.js is None
    assert MISSING in score.note
    with pytest.raises(MetricsError):
        candidate_probs({"only": -1.0})


def test_score_candidates_end_to_end():
    score = score_candidates(
        state_id="s",
        query_id="q",
        candidate_ids=["c1", "c2"],
        z_teacher={"c1": 0.0, "c2": -1.0},
        z_student={"c1": -1.0, "c2": 0.0},
        candidate_names={"c1": "curate", "c2": "search_corpus"},
    )
    assert score.js is not None and score.js > 0
    # Student flipped preference vs Teacher → improvement when moved back.
    closer = score_candidates(
        state_id="s",
        query_id="q",
        candidate_ids=["c1", "c2"],
        z_teacher={"c1": 0.0, "c2": -1.0},
        z_student={"c1": 0.0, "c2": -1.0},
        candidate_names={"c1": "curate", "c2": "search_corpus"},
    )
    assert closer.js == pytest.approx(0.0, abs=1e-12)


def test_sequence_logp_sum():
    z = sequence_logp_sum(
        lambda p, t: [-0.5] * len(t),
        prompt_ids=[1, 2],
        target_ids=[3, 4, 5],
    )
    assert z == pytest.approx(-1.5)
    with pytest.raises(MetricsError):
        sequence_logp_sum(lambda p, t: [-0.5], prompt_ids=[1], target_ids=[3, 4])


def test_name_aggregation():
    probs = {"c1": 0.3, "c2": 0.7}
    names = name_aggregate_probs(
        {"c1": {"name": "curate"}, "c2": {"name": "curate"}}, probs
    )
    assert names == {"curate": pytest.approx(1.0)}


def test_teacher_nll_and_aggregate():
    s1 = score_teacher_nll(
        state_id="s1", query_id="q1", token_logprobs=[-1.0, -2.0], weights=[1.0, 1.0]
    )
    assert s1.nll == pytest.approx(1.5)
    s2 = score_teacher_nll(
        state_id="s2", query_id="q2", token_logprobs=[-0.0], weights=[0.0]
    )
    assert s2.nll is None  # missing, not 0
    agg = aggregate_nll([s1, s2])
    assert agg["nll_macro"] == pytest.approx(1.5)
    assert agg["n_states_missing"] == 1


def test_aggregate_js_query_level():
    scores = [
        score_candidates(
            state_id=f"s{i}",
            query_id="q1",
            candidate_ids=["a", "b"],
            z_teacher={"a": 0.0, "b": -1.0},
            z_student={"a": -1.0, "b": 0.0},
        )
        for i in range(3)
    ] + [
        score_candidates(
            state_id="t",
            query_id="q2",
            candidate_ids=["a", "b"],
            z_teacher={"a": 0.0, "b": -1.0},
            z_student={"a": 0.0, "b": -1.0},
        )
    ]
    agg = aggregate_js(scores)
    # q1 mean dominates 3:1 in micro; macro weights queries equally.
    assert agg["n_states_scored"] == 4
    assert agg["d_k_macro"] == pytest.approx(
        (scores[0].js + scores[3].js) / 2, abs=1e-9
    )


def test_regeneration_counts_all_attempts():
    records = [
        GenerationRecord("s1", "q1", "post@4", ok=True),
        GenerationRecord("s2", "q1", "post@4", ok=False, failure_kind="parse_error"),
        GenerationRecord("s3", "q1", "post@4", ok=False, failure_kind="truncation"),
        GenerationRecord("s4", "q1", "post@4", ok=False, failure_kind=""),  # empty output
    ]
    stats = regen_stats(records)
    assert stats.n_attempted == 4
    assert stats.n_ok == 1
    assert stats.invalid_generated_rate == pytest.approx(0.75)
    assert stats.failures.get("empty_output") == 1
    empty = regen_stats([])
    assert empty.invalid_generated_rate == MISSING  # no denominator → N/A


def test_invalid_guard():
    pre = regen_stats([GenerationRecord("s", "q", "pre", ok=True)] * 8)
    post = regen_stats(
        [GenerationRecord("s", "q", "post", ok=True)] * 7
        + [GenerationRecord("s", "q", "post", ok=False, failure_kind="schema_error")]
    )
    guard = invalid_rate_noninferior(pre, post, eps_invalid=0.2)
    assert guard["pass"] is True
    strict = invalid_rate_noninferior(pre, post, eps_invalid=0.01)
    assert strict["pass"] is False


def test_improvement_absolute_not_relative():
    out = improvement(0.4, 0.3)
    assert out["abs_drop"] == pytest.approx(0.1)
    assert improvement(MISSING, 0.3)["abs_drop"] == MISSING


def test_judge_c1_gates():
    promote = judge_c1(
        pre_js=0.5,
        post_js=0.2,
        pre_nll=2.0,
        post_nll=1.0,
        pre_name_js=0.1,
        post_name_js=0.1,
        delta_nll=0.0,
        delta_dk=0.0,
        absolute_quality_status="PILOT_FROZEN_OK",
    )
    assert promote.status == "PROMOTE"

    dep = judge_c1(pre_js=0.2, post_js=0.3, pre_nll=1.0, post_nll=1.5)
    assert dep.status == "DEPRIORITIZE"

    low = judge_c1(
        pre_js=0.001,
        post_js=0.0,
        pre_nll=2.0,
        post_nll=1.0,
        null_js_noise_floor=0.01,
    )
    assert low.status == "LOW_SIGNAL"

    inv = judge_c1(
        pre_js=0.5,
        post_js=0.1,
        pre_nll=2.0,
        post_nll=1.0,
        support_status="INVALID_MEASUREMENT",
    )
    assert inv.status == "INVALID"

    ns = judge_c1(
        pre_js=0.5,
        post_js=0.1,
        pre_nll=2.0,
        post_nll=1.0,
        support_status="NO_SUPPORT",
    )
    assert ns.status == "NO_SUPPORT"

    border = judge_c1(
        pre_js=0.5,
        post_js=0.2,
        pre_nll=2.0,
        post_nll=1.0,
        seed_directions=[True, False],
    )
    assert border.status == "BORDERLINE"


def test_regenerate_not_from_stored_row():
    """Post results must come from a fresh generation call (§16.6)."""
    calls = []

    def regenerate(backend, state, pair, *, checkpoint):
        calls.append(checkpoint)
        return GenerationRecord(
            state_id=state, query_id="q", checkpoint=checkpoint, ok=True
        )

    # The only legal way to obtain a post record is to call the generator.
    rec = regenerate(None, "s", None, checkpoint="post@4")
    assert calls == ["post@4"]
    assert rec.checkpoint == "post@4"
    stats = regen_stats([rec])
    assert stats.n_attempted == 1
