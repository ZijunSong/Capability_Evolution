from __future__ import annotations

import pytest

from scape.training.batched_env_rollout import rollout_queries_batched
from scape.training.vllm_hybrid import GenerateRequest, GenerateResult, mean_behavior_logprob


def test_batched_rollout_one_generate_call_per_turn():
    pytest.importorskip("openai_harmony")
    from unittest.mock import patch

    calls: list[int] = []

    def generate_batch(requests: list[GenerateRequest]) -> list[GenerateResult]:
        calls.append(len(requests))
        out = []
        for req in requests:
            text = (
                "<|channel|>analysis<|message|>search now.<|end|>"
                "<|start|>assistant to=functions.search_corpus<|channel|>commentary "
                "<|constrain|>json<|message|>{\"query\": \"Apple FY2023\"}<|call|>"
            )
            ids = [1, 2, 200012]
            out.append(
                GenerateResult(
                    request_id=req.request_id,
                    token_ids=ids,
                    token_logprobs=[-0.1, -0.2, -0.3],
                    text=text,
                    logprob_old=mean_behavior_logprob([-0.1, -0.2, -0.3]),
                    logprob_provenance="vllm_sampled_token",
                )
            )
        return out

    rows = [
        {
            "query_id": "q1",
            "query": "When was the Apple FY2023 10-K filed?",
            "gold_docids": ["d1"],
            "frozen_doc_store": {
                "d1": {"id": "d1", "text": "Apple FY2023 10-K was filed on November 3, 2023."},
                "d2": {"id": "d2", "text": "unrelated weather notes"},
            },
        },
        {
            "query_id": "q2",
            "query": "What is the capital of France?",
            "gold_docids": ["d3"],
            "frozen_doc_store": {
                "d3": {"id": "d3", "text": "Paris is the capital of France."},
            },
        },
    ]
    with patch("scape.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]):
        groups = rollout_queries_batched(
            generate_batch,
            rows,
            component_id="sentence_compress",
            group_size=2,
            max_turns=2,
            max_new=32,
            policy_version="v0",
            seed=42,
            sample=True,
            enc=None,
            searcher=None,
        )
    assert len(groups) == 2
    # 2 queries × group 2 = 4 live episodes, both turns unless ended.
    assert calls[0] == 4
    assert all(g.policy_version == "v0" for g in groups)
    rl_rows = list((groups[0].trajectory_group or {}).get("rl_rows") or [])
    assert rl_rows
    assert rl_rows[0]["logprob_provenance"] == "vllm_sampled_token"
    assert rl_rows[0]["action_ids"][-1] == 200012
    assert "token_logprobs" in rl_rows[0]
    assert "action_mask" in rl_rows[0]
    assert groups[0].decision_points[0].policy_version == "v0"
