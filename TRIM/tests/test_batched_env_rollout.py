from __future__ import annotations

import pytest

from trim.training.batched_env_rollout import rollout_queries_batched
from trim.training.vllm_hybrid import GenerateRequest, GenerateResult, mean_behavior_logprob


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
    with patch("trim.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]):
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


def _fake_search_action_result(req: GenerateRequest) -> GenerateResult:
    text = (
        "<|channel|>analysis<|message|>search now.<|end|>"
        "<|start|>assistant to=functions.search_corpus<|channel|>commentary "
        "<|constrain|>json<|message|>{\"query\": \"Apple FY2023\"}<|call|>"
    )
    ids = [1, 2, 200012]
    return GenerateResult(
        request_id=req.request_id,
        token_ids=ids,
        token_logprobs=[-0.1, -0.2, -0.3],
        text=text,
        logprob_old=mean_behavior_logprob([-0.1, -0.2, -0.3]),
        logprob_provenance="vllm_sampled_token",
    )


def _frozen_row(qid: str) -> dict:
    return {
        "query_id": qid,
        "query": f"question {qid}",
        "gold_docids": [f"d-{qid}"],
        "frozen_doc_store": {
            f"d-{qid}": {"id": f"d-{qid}", "text": f"answer text for {qid}"},
        },
    }


def test_resolved_query_batch_size_keeps_about_256_live_episodes():
    from trim.training.batched_env_rollout import resolved_query_batch_size

    assert resolved_query_batch_size(3453, 8, None) == 32
    assert resolved_query_batch_size(830, 1, None) == 256
    assert resolved_query_batch_size(10, 8, None) == 10
    assert resolved_query_batch_size(3453, 8, 0) == 3453
    assert resolved_query_batch_size(3453, 8, 16) == 16


def test_query_batch_submits_vllm_before_all_doc_stores_ready():
    pytest.importorskip("openai_harmony")
    import threading
    from unittest.mock import patch

    from trim.training.four_cell_runtime import doc_store_for_row as real_doc_store

    prepared: list[str] = []
    first_gen_prepared: list[list[str]] = []
    release_later = threading.Event()
    rows = [_frozen_row(f"q{i}") for i in range(6)]

    def wrapped(row, searcher, k=12):
        qid = str(row["query_id"])
        if qid not in {"q0", "q1"}:
            assert release_later.wait(timeout=5.0), "vLLM did not start before later chunks"
        store = real_doc_store(row, searcher, k=k)
        prepared.append(qid)
        return store

    def generate_batch(requests: list[GenerateRequest]) -> list[GenerateResult]:
        if not first_gen_prepared:
            first_gen_prepared.append(list(prepared))
            release_later.set()
        return [_fake_search_action_result(req) for req in requests]

    with (
        patch("trim.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]),
        patch("trim.training.four_cell_runtime.doc_store_for_row", wrapped),
    ):
        groups = rollout_queries_batched(
            generate_batch,
            rows,
            component_id="sentence_compress",
            group_size=1,
            max_turns=1,
            max_new=32,
            policy_version="v0",
            seed=1,
            sample=True,
            enc=None,
            searcher=None,
            query_batch_size=2,
            doc_store_workers=2,
        )

    assert [g.query_id for g in groups] == [f"q{i}" for i in range(6)]
    assert first_gen_prepared
    assert set(first_gen_prepared[0]) == {"q0", "q1"}
    assert set(prepared) == {f"q{i}" for i in range(6)}


def test_doc_store_for_row_unit_cache_and_k_invalidation():
    from trim.eval.browsecomp_retrieval import SearchHit
    from trim.training.four_cell_runtime import doc_store_for_row

    class CountingSearcher:
        name = "count"

        def __init__(self):
            self.n = 0

        def search(self, query, k=5):
            self.n += 1
            return [SearchHit("d1", f"text for {query} k={k}", 1.0)]

    searcher = CountingSearcher()
    row = {"query_id": "q", "query": "apple 10-k"}
    a = doc_store_for_row(row, searcher, k=12)
    b = doc_store_for_row(row, searcher, k=12)
    assert searcher.n == 1
    assert a["d1"]["text"] == b["d1"]["text"]
    c = doc_store_for_row(row, searcher, k=24)
    assert searcher.n == 2
    assert "k=24" in c["d1"]["text"]


def test_doc_store_for_row_live_lucene_empty_does_not_synthesize_gold():
    from trim.training.four_cell_runtime import doc_store_for_row, labeled_doc_store

    class EmptyLucene:
        name = "pyserini_lucene"

        def search(self, query, k=5):
            del query, k
            return []

    row = {"query_id": "q", "query": "apple 10-k", "gold_docids": ["15367"]}
    store = doc_store_for_row(row, EmptyLucene(), k=12)
    assert store == {}
    synthetic = labeled_doc_store(row)
    assert "15367" in synthetic
    assert "15367" not in store
