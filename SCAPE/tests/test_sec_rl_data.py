from __future__ import annotations

import json
import tarfile
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scape.eval.official_query_pool import load_bcplus_830_full
from scape.eval.sec_corpus import (
    SEC_CORPUS_NAME,
    SEC_TRAIN_POOL_NAME,
    attach_sec_doc_stores,
    load_sec_rl_queries,
    normalize_sec_rl_record,
    open_sec_retrieval,
)
from scape.training.four_cell_runtime import resolve_queries, uses_bcplus_830_eval


def _write_rl_tar(tmp: Path, rows: list[dict]) -> Path:
    inner = tmp / "pack"
    inner.mkdir()
    path = inner / "rl_queries_compact.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    archive = tmp / "harness-1-rl-data.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(path, arcname="harness-1-rl-data/rl_queries_compact.jsonl")
    return archive


def _write_corpus(tmp: Path) -> Path:
    root = tmp / "harness-1-sec-corpus"
    shard_dir = root / "corpora" / "sec" / "train"
    shard_dir.mkdir(parents=True)
    table = pa.table(
        {
            "chunk_id": ["chunk_a", "chunk_b"],
            "document_text": ["alpha filing about a director", "the option window lasts 45 days"],
        }
    )
    pq.write_table(table, shard_dir / "train-00000.parquet")
    return root


def test_normalize_sec_rl_record_reads_final_answer_chunks():
    rec = normalize_sec_rl_record(
        {
            "query_id": "0_0",
            "query": "How many days?",
            "answer": "45",
            "document_ids": [
                {"fact": "id", "chunk_ids": ["chunk_a"], "is_final_answer": False},
                {"fact": "ans", "chunk_ids": ["chunk_b", "chunk_c"], "is_final_answer": True},
            ],
        }
    )
    assert rec is not None
    assert rec["gold_docids"] == ["chunk_b", "chunk_c"]
    assert rec["evidence_docids"] == ["chunk_a", "chunk_b", "chunk_c"]
    assert rec["answer"] == "45"
    assert rec["train_pool"] == SEC_TRAIN_POOL_NAME


def test_load_sec_rl_queries_from_tar_and_corpus_lookup(tmp_path: Path):
    rows_in = [
        {
            "query_id": "0_0",
            "query": "How many days comprised this limited window?",
            "answer": "45",
            "document_ids": [
                {"fact": "id", "chunk_ids": ["chunk_a"], "is_final_answer": False},
                {"fact": "ans", "chunk_ids": ["chunk_b"], "is_final_answer": True},
            ],
            "stage": "rl",
        },
        {
            "query_id": "1_0",
            "query": "What is the city?",
            "answer": "Boca Raton",
            "document_ids": [{"fact": "loc", "chunk_ids": ["chunk_a"], "is_final_answer": True}],
            "stage": "rl",
        },
    ]
    archive = _write_rl_tar(tmp_path, rows_in)
    corpus = _write_corpus(tmp_path)
    rows, meta = load_sec_rl_queries(archive, corpus_root=corpus)
    assert len(rows) == 2
    assert rows[0]["gold_docids"] == ["chunk_b"]
    assert meta["pool_contract"] == SEC_TRAIN_POOL_NAME
    assert meta["corpus_name"] == SEC_CORPUS_NAME
    assert meta["corpus_parquet_present"] is True
    assert meta["using_full_train_split"] is True
    seeded = attach_sec_doc_stores(rows, corpus_root=corpus)
    assert seeded["n_rows_seeded"] == 2
    assert "chunk_b" in rows[0]["seed_doc_store"]
    assert "45 days" in rows[0]["seed_doc_store"]["chunk_b"]["text"]
    backend = open_sec_retrieval(corpus, texts={k: v["text"] for row in rows for k, v in row["seed_doc_store"].items()})
    hits = backend.search("option window 45 days", 2)
    assert hits and hits[0].docid == "chunk_b"


def test_load_sec_rl_queries_caps_n(tmp_path: Path):
    archive = _write_rl_tar(
        tmp_path,
        [
            {"query_id": "a", "query": "q1", "document_ids": [{"chunk_ids": ["x"], "is_final_answer": True}]},
            {"query_id": "b", "query": "q2", "document_ids": [{"chunk_ids": ["y"], "is_final_answer": True}]},
        ],
    )
    rows, meta = load_sec_rl_queries(archive, n_queries=1)
    assert [r["query_id"] for r in rows] == ["a"]
    assert meta["query_count_available"] == 2
    assert meta["using_full_train_split"] is False


def test_resolve_queries_scape_rl_uses_sec_train_and_bcplus_830(tmp_path: Path):
    archive = _write_rl_tar(
        tmp_path,
        [
            {
                "query_id": "0_0",
                "query": "How many days?",
                "answer": "45",
                "document_ids": [{"chunk_ids": ["chunk_b"], "is_final_answer": True}],
            }
        ],
    )
    corpus = _write_corpus(tmp_path)
    args = Namespace(
        training_mode="scape_rl",
        query_manifest=None,
        n_queries=None,
        rl_data=archive,
        sec_corpus_root=corpus,
        validate_only=False,
        dry_run=False,
        train_states=None,
        n_train_states=None,
        component="sentence_compress",
        score_split="bcplus_830",
    )
    train_rows, eval_rows, meta, frozen = resolve_queries(args)
    assert [r["query_id"] for r in train_rows] == ["0_0"]
    assert train_rows[0]["gold_docids"] == ["chunk_b"]
    assert train_rows[0]["seed_doc_store"]["chunk_b"]["text"]
    assert len(eval_rows) == 830
    assert {r["official_split"] for r in eval_rows} == {"train", "test"}
    assert sum(1 for r in eval_rows if r["official_split"] == "test") == 166
    assert meta["train"]["pool_contract"] == SEC_TRAIN_POOL_NAME
    assert meta["eval"]["score_split"] == "bcplus_830"
    assert frozen == []


def test_bcplus_830_full_is_664_plus_166():
    rows, meta = load_bcplus_830_full()
    assert len(rows) == 830
    assert meta["score_split"] == "bcplus_830"
    assert meta["query_count_total"] == 830
    assert sum(1 for r in rows if r["official_split"] == "train") == 664
    assert sum(1 for r in rows if r["official_split"] == "test") == 166


def test_uses_bcplus_830_eval_for_scape_rl():
    assert uses_bcplus_830_eval(Namespace(score_split="bcplus_830", training_mode="rl_opd")) is True
    assert uses_bcplus_830_eval(Namespace(score_split="bcplus_full", training_mode="rl_opd")) is True
    assert uses_bcplus_830_eval(Namespace(score_split="bcplus_test_166", training_mode="scape_rl")) is False
    assert uses_bcplus_830_eval(Namespace(score_split=None, training_mode="scape_rl")) is True


def test_official_rl_tar_if_present():
    path = Path("/data/ppnm/harness-1-rl-data.tar.gz")
    if not path.is_file():
        pytest.skip("official harness-1-rl-data.tar.gz not on this machine")
    rows, meta = load_sec_rl_queries(path, n_queries=3)
    assert len(rows) == 3
    assert meta["query_count_available"] == 3453
    assert rows[0]["query"] and rows[0]["query"] != rows[0]["query_id"]
    assert rows[0]["gold_docids"]
