"""Tests for BrowseComp+ BM25 retrieval backend and tools."""

from __future__ import annotations

from unittest.mock import MagicMock

from harness.retrieval.bm25_backend import BrowseCompBm25Backend
from harness.retrieval.bm25_tools import Bm25SearchCorpusTool


class _FakeHit:
    def __init__(self, docid: str, score: float, text: str):
        self.docid = docid
        self.score = score
        self.lucene_document = {"raw": '{"contents": "' + text + '"}'}


class _FakeSearcher:
    def __init__(self):
        self._docs = {
            "42": '{"contents": "Albert Einstein developed relativity."}',
            "99": '{"contents": "Apple Inc. released the iPhone."}',
        }

    def search(self, query: str, k: int):
        if "einstein" in query.lower():
            return [_FakeHit("42", 1.0, "Albert Einstein developed relativity.")]
        return [_FakeHit("99", 0.5, "Apple Inc. released the iPhone.")]

    def doc(self, docid: str):
        raw = self._docs.get(docid)
        if raw is None:
            return None
        return MagicMock(raw=lambda: raw)


def test_bm25_search_tool_returns_formatted_docs(monkeypatch):
    backend = BrowseCompBm25Backend.__new__(BrowseCompBm25Backend)
    backend._searcher = _FakeSearcher()
    tool = Bm25SearchCorpusTool(backend, display_limit=5)
    text, meta = tool({"query": "Who was Einstein?"})
    assert "DOCUMENT ID: 42" in text
    assert meta is not None
    assert meta.returned_chunk_ids == ["42"]


def test_bm25_backend_get_document(monkeypatch):
    backend = BrowseCompBm25Backend.__new__(BrowseCompBm25Backend)
    backend._searcher = _FakeSearcher()
    doc = backend.get_document("42")
    assert doc is not None
    assert "Einstein" in doc
