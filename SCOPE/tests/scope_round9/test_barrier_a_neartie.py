"""Barrier A near-tie agreement for HF↔vLLM float noise."""

from training.scope_round9.aggregate_frozen_replay import (
    _ops_agree_for_barrier,
    barrier_a_for_parity,
    compare_hf_vllm,
)


def test_near_tie_continue_rollback_counts_as_barrier_agree():
    hr = {
        "pred_operation": "ROLLBACK_TO",
        "hf_logits": {"CONTINUE": -5.874, "REPLAN": -11.7, "ROLLBACK_TO": -5.8739},
        "prompt_sha256": "a",
        "token_ids_sha256": "b",
        "candidate_list_sha256": "c",
        "candidate_list": [],
    }
    vr = {
        "pred_operation": "CONTINUE",
        "vllm_logits": {"CONTINUE": -5.828, "REPLAN": -11.8, "ROLLBACK_TO": -5.870},
        "prompt_sha256": "a",
        "token_ids_sha256": "b",
        "candidate_list_sha256": "c",
        "candidate_list": [],
    }
    raw, bar = _ops_agree_for_barrier(hr, vr, near_tie_eps=0.1)
    assert raw is False
    assert bar is True


def test_clear_disagreement_not_forgiven():
    hr = {
        "pred_operation": "ROLLBACK_TO",
        "hf_logits": {"CONTINUE": -9.0, "REPLAN": -12.0, "ROLLBACK_TO": -5.0},
        "prompt_sha256": "a",
        "token_ids_sha256": "b",
        "candidate_list_sha256": "c",
        "candidate_list": [],
    }
    vr = {
        "pred_operation": "CONTINUE",
        "vllm_logits": {"CONTINUE": -3.0, "REPLAN": -12.0, "ROLLBACK_TO": -8.0},
        "prompt_sha256": "a",
        "token_ids_sha256": "b",
        "candidate_list_sha256": "c",
        "candidate_list": [],
    }
    raw, bar = _ops_agree_for_barrier(hr, vr, near_tie_eps=0.1)
    assert raw is False
    assert bar is False


def test_compare_reports_barrier_agreement_1():
    hf = [
        {
            "pred_operation": "CONTINUE",
            "hf_logits": {"CONTINUE": -5.0, "REPLAN": -10.0, "ROLLBACK_TO": -5.05},
            "prompt_sha256": "p",
            "token_ids_sha256": "t",
            "candidate_list_sha256": "c",
            "candidate_list": [{"local_checkpoint_id": "C0"}],
        }
    ]
    vl = [
        {
            "pred_operation": "ROLLBACK_TO",
            "vllm_logits": {"CONTINUE": -5.08, "REPLAN": -10.0, "ROLLBACK_TO": -5.0},
            "prompt_sha256": "p",
            "token_ids_sha256": "t",
            "candidate_list_sha256": "c",
            "candidate_list": [{"local_checkpoint_id": "C0"}],
            "pred_checkpoint_local_id": "C0",
        }
    ]
    p = compare_hf_vllm(hf, vl)
    assert p["operation_top1_agreement"] == 1.0
    assert p["near_tie_resolved_count"] == 1
    ok, fails = barrier_a_for_parity(p)
    assert ok and fails == []
