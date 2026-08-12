"""A12 hierarchical rollback tests."""

from __future__ import annotations

from experiments.ablations.builders.build_rollback_hierarchy import (
    offline_hard_gate,
    predict_hierarchical,
)


def test_oracle_path():
    state = {
        "gold_operation": "ROLLBACK",
        "gold_checkpoint_id": "ck1",
        "candidates": ["ck1", "ck2"],
    }
    hp = predict_hierarchical(
        state,
        variant="a12_oracle_operation_oracle_checkpoint",
        op_model=lambda s: "CONTINUE",
        ckpt_ranker=lambda s: "ck2",
        oracle_operation=lambda s: s["gold_operation"],
        oracle_checkpoint=lambda s: s["gold_checkpoint_id"],
        executability_check=lambda c: c == "ck1",
    )
    assert hp.operation == "ROLLBACK"
    assert hp.checkpoint_id == "ck1"
    assert hp.executable is True


def test_hard_gate_blocks_weak_metrics():
    g = offline_hard_gate(
        {
            "operation_type_balanced_accuracy": 0.5,
            "CONTINUE_recall": 0.5,
            "checkpoint_selection_accuracy": 0.4,
            "invalid_checkpoint_rate": 0.1,
            "post_action_invariant_pass_rate": 0.9,
            "hf_vllm_operation_parity": 0,
            "seed_direction_consistent": False,
        }
    )
    assert g["passed"] is False
    assert g["allow_100q_closed_loop"] is False
