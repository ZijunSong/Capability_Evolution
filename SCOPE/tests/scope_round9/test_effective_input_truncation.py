"""Effective-input truncation must keep the decision head suffix."""

from transformers import AutoTokenizer

from training.scope.rollback_effective_input import build_rollback_effective_input


def test_truncation_keeps_recovery_operation_suffix():
    tok = AutoTokenizer.from_pretrained(
        "/data/ppnm/models/Qwen2.5-7B-Instruct", trust_remote_code=True
    )
    long_ctx = ("Evidence blob " + ("word " * 5000)).strip()
    sample = {
        "query_id": "q1",
        "turn_id": 3,
        "student_state_text": long_ctx,
        "decision_state": {
            "turn_id": 3,
            "available_checkpoints": [
                {
                    "checkpoint_id": "ckpt_1_aaa",
                    "turn_id": 1,
                    "n_curated": 1,
                    "n_pool": 2,
                }
            ],
        },
        "target_action": {"operation": "CONTINUE"},
    }
    eff = build_rollback_effective_input(sample, tok, max_length=256)
    assert eff.truncated
    assert eff.effective_input_text.rstrip().endswith("Recovery operation:")
    assert eff.token_length_after <= 256
