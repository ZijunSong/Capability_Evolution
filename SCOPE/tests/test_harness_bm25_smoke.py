"""Offline smoke tests for Harness + BM25 (no API keys / Java / Lucene)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.harness_config import apply_harness_config, config_path, load_harness_config
from harness.retrieval.bm25_backend import _is_lucene_index_dir, resolve_bm25_index_path
from harness.retrieval.bm25_tools import (
    Bm25GrepCorpusTool,
    Bm25ReadDocumentTool,
    Bm25SearchCorpusTool,
)
from harness.retrieval.memory_backend import InMemoryBm25Backend
from training.opd.env_factory import build_search_env, build_smoke_bm25_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend


@pytest.fixture
def memory_backend() -> InMemoryBm25Backend:
    return InMemoryBm25Backend()


def test_lucene_index_dir_ignores_lock_files(tmp_path: Path):
    idx = tmp_path / "bm25"
    idx.mkdir()
    (idx / "segments_3.lock").write_text("x", encoding="utf-8")
    assert not _is_lucene_index_dir(idx)

    (idx / "segments_1").write_text("x", encoding="utf-8")
    assert _is_lucene_index_dir(idx)


def test_resolve_bm25_index_path_skips_dot_cache(tmp_path: Path):
    root = tmp_path / "indexes"
    cache = root / ".cache" / "hf" / "bm25"
    cache.mkdir(parents=True)
    (cache / "segments_1.lock").write_text("x", encoding="utf-8")

    real = root / "bm25"
    real.mkdir()
    (real / "segments_1").write_text("x", encoding="utf-8")

    resolved = resolve_bm25_index_path(root / "bm25")
    assert resolved == real.resolve()


def test_smoke_retrieval_preflight():
    assert check_retrieval_backend("bm25", smoke=True) == "memory://smoke"


def test_smoke_toolstack_search_grep_read(memory_backend: InMemoryBm25Backend):
    search = Bm25SearchCorpusTool(memory_backend, display_limit=3)
    text, meta = search({"query": "Einstein relativity"})
    assert "1001" in text
    assert meta is not None

    grep = Bm25GrepCorpusTool(memory_backend)
    grep_text, grep_meta = grep({"pattern": "Macintosh"})
    assert "1002" in grep_text
    assert grep_meta is not None

    read = Bm25ReadDocumentTool(memory_backend, token_counter=lambda t: len(t.split()))
    doc, _ = read({"doc_id": "1003"})
    assert "Perseverance" in doc


def test_smoke_rollout_runtime_and_env():
    cfg = load_harness_config(config_path("modules_full.yaml"))
    apply_harness_config(cfg)

    runtime = build_smoke_bm25_rollout_runtime()
    assert runtime.retrieval_backend == "bm25"
    assert len(runtime.toolset.tools) >= 4

    fixture = Path(__file__).parent / "fixtures" / "browsecomp_sample_queries.json"
    record = json.loads(fixture.read_text(encoding="utf-8"))[0]
    env = build_search_env(
        runtime,
        query_id=record["query_id"],
        query_text=record["query"],
        max_turns=2,
    )
    assert env.wm.query == record["query"]


def test_chroma_preflight_rejects_example_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "EXAMPLE")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "EXAMPLE")
    monkeypatch.setenv("CHROMA_API_KEY", "EXAMPLE")
    monkeypatch.setenv("CHROMA_DATABASE", "EXAMPLE")
    monkeypatch.setenv("TINKER_API_KEY", "EXAMPLE")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "EXAMPLE")
    monkeypatch.setenv("MOONSHOT_API_KEY", "EXAMPLE")
    monkeypatch.setenv("BASETEN_API_KEY", "EXAMPLE")
    monkeypatch.setenv("BASETEN_MODEL_URL", "EXAMPLE")
    monkeypatch.setenv("JINA_API_KEY", "EXAMPLE")
    monkeypatch.setenv("CONTEXTUAL_API_KEY", "EXAMPLE")
    monkeypatch.setenv(
        "BROWSECOMPPLUS_QUERIES_PATH",
        str(Path(__file__).resolve().parents[1] / "external/BrowseComp-Plus/topics-qrels/queries.tsv"),
    )
    monkeypatch.setenv(
        "BROWSECOMPPLUS_QRELS_GOLD_PATH",
        str(Path(__file__).resolve().parents[1] / "external/BrowseComp-Plus/topics-qrels/qrel_golds.txt"),
    )
    monkeypatch.setenv(
        "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH",
        str(Path(__file__).resolve().parents[1] / "external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt"),
    )
    monkeypatch.setenv(
        "BROWSECOMPPLUS_ANSWERS_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"
        ),
    )

    from harness.config import get_config

    get_config.cache_clear()

    with pytest.raises(RuntimeError, match="CHROMA_API_KEY"):
        check_retrieval_backend("chroma")
