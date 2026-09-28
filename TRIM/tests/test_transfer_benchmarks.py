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
    LocalQueryPoolDataset,
    open_eval_retrieval,
    parent_chunk_id,
    resolve_local_bm25_corpus,
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
        assert args.retrieval_backend == "local_bm25"
        assert args.reranker == "none"
        assert args.offline is True
        assert args.upstream_dataset == canon
        assert args.index_path.endswith(f"{canon}/indexes/bm25")
        assert args.corpus_path.endswith(f"{canon}/corpus.jsonl")


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


def test_load_eval_benchmark_bcplus_test_50():
    from trim.eval.official_query_pool import SCORE_SPLIT_50, load_bcplus_830_split

    rows, meta = load_eval_benchmark("bcplus_test_50")
    _train, test166, _ = load_bcplus_830_split()
    assert len(rows) == 50
    assert meta["score_split"] == SCORE_SPLIT_50
    assert meta["subset_of"] == "bcplus_test_166"
    assert [r["query_id"] for r in rows] == [r["query_id"] for r in test166[:50]]
    assert all(r["official_split"] == "test" for r in rows)
    args, spec = parse_eval_args(
        ["--benchmark", "bcplus_test_50", "--component", "zero", "--out", "/tmp/trim-eval-50"]
    )
    assert spec.benchmark == "bcplus_test_50"
    assert args.score_split == "bcplus_test_50"
    assert args.retrieval_backend == "local_bm25"
    assert args.reranker == "none"
    assert args.upstream_dataset == "browsecompplus"
    assert args.index_path.endswith("indexes/bm25")
    assert args.corpus_path.endswith("browsecomp_plus_corpus_full.jsonl") or args.corpus_path.endswith(
        "browsecomp_plus_corpus.jsonl"
    )


def test_eval_cli_explicit_chroma_is_opt_in(tmp_path: Path):
    args, spec = parse_eval_args(
        [
            "--benchmark",
            "longsealqa",
            "--component",
            "zero",
            "--out",
            str(tmp_path / "chroma"),
            "--retrieval-backend",
            "upstream",
        ]
    )
    assert spec.benchmark == "longsealqa"
    assert args.retrieval_backend == "upstream"
    assert args.index_path is None


def test_eval_cli_explicit_index_wins(tmp_path: Path):
    args, _spec = parse_eval_args(
        [
            "--benchmark",
            "frames",
            "--component",
            "zero",
            "--out",
            str(tmp_path / "frames-custom"),
            "--index-path",
            "/tmp/custom-index",
            "--corpus-path",
            "/tmp/custom-corpus.jsonl",
        ]
    )
    assert args.retrieval_backend == "local_bm25"
    assert args.index_path == "/tmp/custom-index"
    assert args.corpus_path == "/tmp/custom-corpus.jsonl"
    assert args.upstream_dataset == "frames"


def test_resolve_local_bm25_corpus_matches_built_transfer_dirs():
    for name in ("longsealqa", "frames", "hotpotqa"):
        corpus = resolve_local_bm25_corpus(name)
        assert corpus.benchmark == name
        assert corpus.dataset == name
        assert corpus.index_path.as_posix().endswith(f"{name}/indexes/bm25")
        assert corpus.corpus_path.name == "corpus.jsonl"
        if not corpus.index_path.is_dir() or not corpus.corpus_path.is_file():
            continue
        assert corpus.index_path.is_dir()
        assert corpus.corpus_path.is_file()


def test_local_query_pool_dataset_matches_chunk_parent():
    ds = LocalQueryPoolDataset(
        "longsealqa",
        [
            {
                "query_id": "longseal-000",
                "query": "Which gold document mentions Brussels?",
                "answer": "1878",
                "gold_docids": ["https://example.com/gold"],
                "evidence_docids": ["https://example.com/gold"],
            }
        ],
    )
    assert ds.get_query_text("longseal-000") == "Which gold document mentions Brussels?"
    assert ds.evaluate_results_recall("longseal-000", ["https://example.com/gold::c0"]) == 1.0
    assert ds.evaluate_results_final_answer_recall("longseal-000", ["https://example.com/gold::c0"]) == 1.0
    assert ds.evaluate_results_precision("longseal-000", ["https://example.com/gold::c0", "https://other"]) == 0.5
    assert ds.evaluate_results_recall("longseal-000", ["https://other"]) == 0.0
