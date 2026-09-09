"""L03–L06 / L12: local BM25 corpus, ID map, grep/read contract, no Chroma fallback."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from trim.local_backend.corpus_store import LocalCorpusStore
from trim.local_backend.format_obs import DOCUMENT_ID_PREFIX, format_search_observation
from trim.local_backend.id_map import IdMap
from trim.local_backend.tools import LocalGrepCorpusTool, LocalReadDocumentTool, LocalSearchCorpusTool
from trim.upstream_harness1.retrieval import (
    RETRIEVAL_LOCAL_BM25,
    RETRIEVAL_UPSTREAM,
    PROFILE_LOCAL_BM25,
    RetrievalConfig,
    retrieval_from_args,
)


@dataclass
class FakeSchema:
    name: str
    description: str = ""
    parameters: dict | None = None

    def model_dump(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters or {}}


@dataclass
class FakeMeta:
    returned_chunk_ids: list
    pre_rerank_chunk_ids: list | None = None


class FakeBm25:
    index_version = "test-index"

    def search(self, query, *, k, ignore_ids=None):
        from trim.local_backend.bm25 import RankHit

        ignore = {str(x) for x in (ignore_ids or [])}
        hits = [
            RankHit("d000000_0", "doc_with_underscore", "alpha token unique_tail_zzz", 1.2),
            RankHit("plain_0", "plain", "plain document body and more", 0.9),
        ]
        return [h for h in hits if h.chunk_id not in ignore and h.official_id not in ignore][:k]


def _store():
    return LocalCorpusStore.from_memory(
        [
            {
                "id": "doc_with_underscore",
                "text": "alpha token unique_tail_zzz " + ("middle " * 20) + "evidence_at_the_end",
            },
            {"id": "plain", "text": "plain document body and more"},
        ]
    )


def test_id_map_keeps_official_ids_with_underscores():
    mapping = IdMap()
    internal = mapping.register_official("doc_with_underscore")
    assert internal != "doc"
    assert mapping.official_of(f"{internal}_0") == "doc_with_underscore"
    assert mapping.official_of("doc_with_underscore") == "doc_with_underscore"
    assert mapping.to_official_list([f"{internal}_0", "plain_0"])[0] == "doc_with_underscore"


def test_formatter_document_marker_matches_upstream():
    text = format_search_observation(["plain_0"], ["hello world"], display_limit=10)
    assert DOCUMENT_ID_PREFIX in text
    assert "# DOCUMENT ID: plain_0\n" in text
    assert "hello world" in text
    assert format_search_observation([], []) == "No results found"


def test_formatter_header_matches_sentence_compress_regex():
    import re

    text = format_search_observation(["123_0"], ["body text"], token_counts=[3], display_limit=10)
    assert re.search(r"(# DOCUMENT ID:\s*\S+\n)", text)
    assert "(3 tokens)" in text
    assert "# DOCUMENT ID: 123_0 (3 tokens)" not in text


def test_query_jsonl_rejected_as_corpus(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"1","query":"hello","answer":"x","gold_docs":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="0 documents"):
        LocalCorpusStore.from_jsonl(path)


def test_grep_real_regex_invalid_timeout_and_empty():
    store = _store()
    grep = LocalGrepCorpusTool(
        store=store,
        metadata_cls=FakeMeta,
        schema=FakeSchema("grep_corpus"),
        timeout_s=5.0,
    )
    text, meta = grep({"pattern": "unique_tail_zzz"})
    assert "unique_tail_zzz" in text
    assert meta.returned_chunk_ids
    bad, bad_meta = grep({"pattern": "("})
    assert "invalid regex" in bad
    assert bad_meta.returned_chunk_ids == []
    empty, empty_meta = grep({"pattern": "definitely_not_in_corpus_abc123"})
    assert empty == "No results found"
    assert empty_meta.returned_chunk_ids == []
    grep._timeout_s = 0.0
    timed, timed_meta = grep({"pattern": "plain"})
    assert "timed out" in timed
    assert timed_meta.returned_chunk_ids == []


def test_read_recovers_tail_and_checksum_is_stable():
    store = _store()
    before = store.checksum_of("doc_with_underscore")
    read = LocalReadDocumentTool(store=store, schema=FakeSchema("read_document"))
    text, _meta = read({"doc_id": "doc_with_underscore"})
    assert "evidence_at_the_end" in text
    assert store.checksum_of("doc_with_underscore") == before
    missing, _ = read({"doc_id": "unknown_id"})
    assert "unknown id" in missing


def test_search_metadata_only_includes_displayed_ids():
    store = _store()
    search = LocalSearchCorpusTool(
        backend=FakeBm25(),
        store=store,
        metadata_cls=FakeMeta,
        schema=FakeSchema("search_corpus"),
        display_limit=1,
        search_limit=10,
    )
    text, meta = search({"query": "alpha"})
    assert meta.returned_chunk_ids == ["d000000_0"]
    assert "d000000_0" in text
    assert search.capability_log["adaptive_rerank_instruction_consumed_by_search"] is False
    _, meta2 = search({"query": "alpha"}, {"rerank_instruction": "prefer recent"})
    assert search.capability_log["adaptive_rerank_instruction_received"] is True
    assert search.capability_log["adaptive_rerank_instruction_consumed_by_search"] is False
    assert meta2.returned_chunk_ids == ["d000000_0"]


def test_local_bm25_requires_index_and_corpus():
    cfg = RetrievalConfig(backend=RETRIEVAL_LOCAL_BM25)
    with pytest.raises(RuntimeError, match="index-path"):
        cfg.assert_ready()
    cfg2 = RetrievalConfig(backend=RETRIEVAL_LOCAL_BM25, index_path="/tmp/idx")
    with pytest.raises(RuntimeError, match="corpus-path"):
        cfg2.assert_ready()
    with pytest.raises(RuntimeError, match="local_hybrid"):
        RetrievalConfig(backend="local_hybrid").assert_ready()
    with pytest.raises(RuntimeError, match="retired"):
        RetrievalConfig(backend="substitute_bm25").assert_ready()


def test_factory_does_not_fall_back_to_jsonl_overlap(tmp_path):
    from trim.local_backend.factory import build_local_toolset

    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id":"a","text":"hello"}\n', encoding="utf-8")
    cfg = RetrievalConfig(
        backend=RETRIEVAL_LOCAL_BM25,
        index_path=str(tmp_path / "missing_index"),
        corpus_path=str(corpus),
        reranker="none",
    )
    with pytest.raises(RuntimeError, match="index_path does not exist"):
        build_local_toolset({}, cfg)


def test_harness_tools_import_does_not_load_chromadb():
    import sys

    from trim.upstream_harness1.pin import HARNESS1_ROOT, ensure_harness1_on_path

    if not HARNESS1_ROOT.is_dir():
        pytest.skip("vendored harness-1 missing")
    ensure_harness1_on_path()
    before = "chromadb" in sys.modules
    try:
        import harness.tools  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(f"upstream harness import deps missing: {exc}")
    if not before:
        assert "chromadb" not in sys.modules
    args = SimpleNamespace(
        retrieval_backend="local_bm25",
        reranker="none",
        index_path="/data/idx",
        corpus_path="/data/corpus.jsonl",
        docstore_path="",
        id_map_path="",
        corpus_manifest="",
        corpus_version="v1",
        verify_base_url="http://127.0.0.1:8001/v1",
        reranker_base_url="",
        reranker_model="",
        offline=True,
        upstream_dataset="browsecomp_plus",
    )
    cfg = retrieval_from_args(args)
    assert cfg.backend == RETRIEVAL_LOCAL_BM25
    assert cfg.eval_profile() == PROFILE_LOCAL_BM25
    assert cfg.dataset == "browsecompplus"
    cfg.assert_local_bm25()
    RetrievalConfig(backend=RETRIEVAL_UPSTREAM).assert_official_baseline()
    with pytest.raises(RuntimeError, match="local_bm25"):
        RetrievalConfig(backend="substitute_bm25").assert_official_baseline()
