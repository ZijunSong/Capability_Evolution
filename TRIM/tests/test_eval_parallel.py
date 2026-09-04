import pytest

from trim.cli.launch import LaunchError, parse_eval_args
from trim.eval.eval_parallel import (
    assign_replica_gpus,
    effective_replica_count,
    merge_traces,
    shard_rows_round_robin,
    summarize_merged_traces,
)


def test_parse_eval_cli_tp():
    args, _spec = parse_eval_args(
        [
            "--benchmark",
            "bcplus_test_166",
            "--component",
            "all",
            "--tp",
            "8",
            "--eval-gpus",
            "0,1,2,3,4,5,6,7",
            "--max-num-seqs",
            "128",
            "--eval-chunk-size",
            "32",
            "--out",
            "/tmp/trim-eval-tp",
        ]
    )
    assert args.eval_replicas == 8
    assert args.eval_gpus == "0,1,2,3,4,5,6,7"
    assert args.max_num_seqs == 128
    assert args.eval_chunk_size == 32
    assert args.eval_stagger_s == 0.0


def test_parse_eval_cli_tp_alias():
    args, _spec = parse_eval_args(
        ["--component", "zero", "--eval-replicas", "4", "--out", "/tmp/trim-eval-replicas"]
    )
    assert args.eval_replicas == 4


def test_parse_eval_cli_tp_must_be_positive():
    with pytest.raises(LaunchError):
        parse_eval_args(["--component", "zero", "--tp", "0", "--out", "/tmp/trim-eval-tp0"])


def test_assign_replica_gpus_one_gpu_each():
    groups = assign_replica_gpus(list(range(8)), n_replicas=8, tp_size=1)
    assert groups == [[i] for i in range(8)]


def test_assign_replica_gpus_with_model_tp():
    groups = assign_replica_gpus(list(range(8)), n_replicas=4, tp_size=2)
    assert groups == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_assign_replica_gpus_rejects_short_pool():
    with pytest.raises(ValueError, match="needs 8 GPUs"):
        assign_replica_gpus([0, 1, 2, 3], n_replicas=8, tp_size=1)


def test_round_robin_shards_cover_the_full_set():
    rows = [{"query_id": f"q{i}", "query": f"Q{i}"} for i in range(166)]
    shards = shard_rows_round_robin(rows, 8)
    assert len(shards) == 8
    assert [len(s) for s in shards] == [21, 21, 21, 21, 21, 21, 20, 20]
    ids = [row["query_id"] for shard in shards for row in shard]
    assert sorted(ids, key=lambda x: int(x[1:])) == [f"q{i}" for i in range(166)]
    assert len(set(ids)) == 166


def test_merge_traces_restores_original_order():
    rows = [{"query_id": f"q{i}"} for i in range(10)]
    shards = shard_rows_round_robin(rows, 4)
    shard_traces = [[{"query_id": r["query_id"], "rank": i} for r in shard] for i, shard in enumerate(shards)]
    merged = merge_traces(shard_traces, rows)
    assert [t["query_id"] for t in merged] == [f"q{i}" for i in range(10)]
    assert {t["query_id"]: t["rank"] for t in merged}["q0"] == 0
    assert {t["query_id"]: t["rank"] for t in merged}["q1"] == 1


def test_effective_replica_count_does_not_exceed_queries():
    assert effective_replica_count(6, 8) == 6
    assert effective_replica_count(166, 8) == 8
    assert effective_replica_count(0, 8) == 1


def test_summarize_merged_traces_averages_full_set():
    rows = [
        {"query_id": "a", "official_split": "test"},
        {"query_id": "b", "official_split": "test"},
    ]
    traces = [
        {"query_id": "a", "tool_names": ["search_corpus"], "evidence_recall_at_5": 1.0, "n_tool_calls": 2, "n_search_calls": 1},
        {"query_id": "b", "tool_names": ["search_corpus"], "evidence_recall_at_5": 0.0, "n_tool_calls": 4, "n_search_calls": 3},
    ]
    summary = summarize_merged_traces(
        traces,
        rows,
        leak_count=0,
        primary_split="bcplus_test_166",
        retrieval_name="pyserini_lucene",
    )
    assert summary["n_queries"] == 2
    assert summary["mean_tool_calls_per_query"] == 3.0
    assert summary["tool_search_cost"] == 2.0
    assert summary["test_evidence_recall_at_5"] == 0.5
    assert summary["teacher_leak_rate"] == 0.0
