#!/usr/bin/env python3
"""Smoke test Harness + BM25 retrieval without API keys or Lucene index."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Block accidental live credential use during smoke.
os.environ.setdefault("OPENAI_API_KEY", "SMOKE_OFFLINE")
os.environ.setdefault("CHROMA_API_KEY", "SMOKE_OFFLINE")

from harness.harness_config import apply_harness_config, config_path, load_harness_config
from harness.retrieval.memory_backend import InMemoryBm25Backend
from training.opd.env_factory import build_smoke_bm25_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend
from training.opd.rollout_worker import QueryRecord


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  OK  {msg}")


def main() -> None:
    print("=== Harness BM25 smoke (no API keys) ===\n")

    print("[1/5] Harness module config")
    cfg = load_harness_config(config_path("modules_full.yaml"))
    apply_harness_config(cfg)
    _assert(cfg.evidence_state.enabled, "modules_full evidence_state enabled")

    print("\n[2/5] Retrieval preflight (in-memory)")
    path = check_retrieval_backend("bm25", smoke=True)
    _assert(path == "memory://smoke", f"smoke backend path = {path}")

    print("\n[3/5] Toolstack build + search/grep/read")
    runtime = build_smoke_bm25_rollout_runtime()
    _assert(runtime.retrieval_backend == "bm25", "runtime uses bm25 backend")

    search_text, search_meta = runtime.search_tool(
        {"query": "Einstein Nobel Prize relativity"}
    )
    _assert("DOCUMENT ID: 1001" in search_text, "search_corpus returns Einstein doc")
    _assert(search_meta is not None and search_meta.returned_chunk_ids, "search metadata")

    grep_tool = runtime.toolset.get_tool("grep_corpus")
    _assert(grep_tool is not None, "grep_corpus tool present")
    grep_text, grep_meta = grep_tool({"pattern": "Jezero"})
    _assert("1003" in grep_text, "grep_corpus finds Perseverance doc")

    read_tool = runtime.toolset.get_tool("read_document")
    _assert(read_tool is not None, "read_document tool present")
    doc_text, _ = read_tool({"doc_id": "1002"})
    _assert("Apple" in doc_text, "read_document returns Apple doc")

    print("\n[4/5] BrowseComp query fixture alignment")
    fixture = _REPO_ROOT / "tests/fixtures/browsecomp_sample_queries.json"
    records = [QueryRecord(**item) for item in json.loads(fixture.read_text())]
    _assert(len(records) == 3, "loaded 3 sample queries")
    for record in records:
        backend = InMemoryBm25Backend()
        hits = backend.search(record.query, k=3)
        _assert(len(hits) > 0, f"BM25 hit for query_id={record.query_id}")

    print("\n[5/5] Env factory search env (no vLLM episode)")
    from training.opd.env_factory import build_search_env

    env = build_search_env(
        runtime,
        query_id=records[0].query_id,
        query_text=records[0].query,
        max_turns=3,
    )
    _assert(env.query_id == records[0].query_id, "SlidingWindowSearchEnv constructed")
    _assert(env.search_tool is runtime.search_tool, "env wired to bm25 search tool")

    print("\n=== All Harness BM25 smoke checks passed ===")


if __name__ == "__main__":
    main()
