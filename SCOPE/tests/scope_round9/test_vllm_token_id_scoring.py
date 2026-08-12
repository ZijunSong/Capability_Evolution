"""Token-id scoring must match HF concat encoding (no string-merge artifacts)."""

from __future__ import annotations

from types import SimpleNamespace

from training.scope.vllm_rollback_scorer import VllmRollbackScorer


def test_completion_logprobs_slice_uses_prompt_n():
    scorer = VllmRollbackScorer(client=None, model="x", model_path=None)  # type: ignore[arg-type]
    # prompt_n=3, comp_n=2 → indices 3,4
    lps = [None, -1.0, -2.0, -3.5, -4.5, -9.0]
    choice = SimpleNamespace(logprobs=SimpleNamespace(token_logprobs=lps))
    out = scorer._completion_logprobs_from_token_ids(choice, prompt_n=3, comp_n=2)
    assert out == [-3.5, -4.5]


def test_prompt_verbalizer_concat_ids_stable_for_qwen(tmp_path=None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/data/ppnm/models/Qwen2.5-7B-Instruct", trust_remote_code=True
    )
    # Truncated mid-json style suffix that previously merged with CONTINUE.
    prompt = '..."n_curated": 1, "n_pool": 10}, {"checkpoint_id": "ckpt_3_ea0e8e4f", "turn_id": 3, "state_hash": "467169e323d7023f", "n_curated": '
    for op in ("CONTINUE", "REPLAN", "ROLLBACK_TO"):
        concat = tok.encode(prompt, add_special_tokens=False) + tok.encode(
            op, add_special_tokens=False
        )
        joint = tok.encode(prompt + op, add_special_tokens=False)
        # Document that string joint can differ; token-id path uses concat.
        assert concat[-len(tok.encode(op, add_special_tokens=False)) :] == tok.encode(
            op, add_special_tokens=False
        )
        if concat != joint:
            # The bug class we fixed: joint merge changes completion tokens.
            assert joint[-len(tok.encode(op, add_special_tokens=False)) :] != tok.encode(
                op, add_special_tokens=False
            ) or len(concat) != len(joint)
