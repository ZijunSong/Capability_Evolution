from __future__ import annotations

import json
from pathlib import Path

from trim.cli.launch import parse_eval_args
from trim.eval.local_search_env import curated_recall
from trim.eval.official_query_pool import score_split_for_benchmark
from trim.eval.transfer_benchmarks import (
    TransferRetrievalBackend,
    canonical_transfer_benchmark,
    load_eval_benchmark,
    load_transfer_queries,
    open_eval_retrieval,
    parent_chunk_id,
    wiki_title_key,
)


def test_cli_transfer_benchmark_aliases(tmp_path: Path):
    for raw, canon in (
        ("longsealqa", "longsealqa"),
        ("LongSeal", "longsealqa"),
        ("frames", "frames"),
        ("hotpotqa_subset", "hotpotqa"),
        ("web", "web"),
        ("patents", "patents"),
    ):
        args, spec = parse_eval_args(
            ["--benchmark", raw, "--component", "zero", "--out", str(tmp_path / canon)]
        )
        assert spec.benchmark == canon
        assert args.score_split == canon


def test_score_split_for_benchmark_transfer():
    assert score_split_for_benchmark("longsealqa") == "longsealqa"
    assert score_split_for_benchmark("frames") == "frames"
    assert score_split_for_benchmark("hotpotqa") == "hotpotqa"
    assert canonical_transfer_benchmark("hotpotqa_subset") == "hotpotqa"


def test_wiki_and_chunk_id_matching():
    assert wiki_title_key("https://en.wikipedia.org/wiki/James_Buchanan") == "James Buchanan"
    assert wiki_title_key("James Buchanan") == "James Buchanan"
    assert parent_chunk_id("https://example.com/gold::c2") == "https://example.com/gold"


def test_curated_recall_matches_chunk_parent():
    state = {"curated": {"https://example.com/gold::c0": {"id": "https://example.com/gold::c0"}}}
    assert curated_recall(state, ["https://example.com/gold"]) == 1.0


def test_load_transfer_queries_from_local_manifest(tmp_path: Path, monkeypatch):
    root = tmp_path / "transfer_local"
    bench = root / "longsealqa"
    bench.mkdir(parents=True)
    rows = [
        {
            "query_id": "longseal-000",
            "query": "Which gold document mentions Brussels?",
            "answer": "1878",
            "gold_docids": ["https://example.com/gold"],
            "evidence_docids": ["https://example.com/gold"],
            "official_split": "test",
        }
    ]
    (bench / "queries.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    (bench / "corpus.jsonl").write_text(
        json.dumps({"id": "https://example.com/gold", "contents": "Brussels synagogue 1878"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRIM_TRANSFER_CORPUS_ROOT", str(root))
    loaded, meta = load_transfer_queries("longsealqa")
    assert len(loaded) == 1
    assert loaded[0]["gold_docids"] == ["https://example.com/gold"]
    assert meta["query_count"] == 1
    eval_rows, eval_meta = load_eval_benchmark("longsealqa")
    assert eval_rows[0]["query_id"] == "longseal-000"
    assert eval_meta["score_split"] == "longsealqa"
    searcher = open_eval_retrieval("longsealqa", formal=False)
    assert isinstance(searcher, TransferRetrievalBackend)
    hits = searcher.search("Brussels synagogue", 3)
    assert hits
    assert searcher.normalize_id(hits[0].docid) == "https://example.com/gold"
