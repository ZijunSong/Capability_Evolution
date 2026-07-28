"""In-memory BM25 backend for smoke tests (no Java / Pyserini / API keys)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.retrieval.bm25_backend import Bm25Hit


@dataclass(frozen=True)
class MemoryDocument:
    doc_id: str
    text: str


_DEFAULT_DOCS: tuple[MemoryDocument, ...] = (
    MemoryDocument(
        "1001",
        "Albert Einstein developed the theory of relativity and won the Nobel Prize in Physics in 1921.",
    ),
    MemoryDocument(
        "1002",
        "Apple Inc., founded in 1976, released the Macintosh, an early personal computer with a graphical user interface.",
    ),
    MemoryDocument(
        "1003",
        "NASA's Perseverance rover landed in Jezero Crater on Mars in February 2021.",
    ),
)


class InMemoryBm25Backend:
    """Tiny lexical backend for offline Harness smoke tests."""

    index_path = "memory://smoke"

    def __init__(self, documents: list[MemoryDocument] | None = None) -> None:
        self._documents = list(documents or _DEFAULT_DOCS)

    def search(
        self,
        query: str,
        *,
        k: int = 50,
        ignore_ids: set[str] | None = None,
    ) -> list[Bm25Hit]:
        if not query.strip():
            return []
        ignore = ignore_ids or set()
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2]
        scored: list[tuple[float, MemoryDocument]] = []
        for doc in self._documents:
            if doc.doc_id in ignore:
                continue
            text_l = doc.text.lower()
            score = sum(1.0 for term in terms if term in text_l)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id))
        return [
            Bm25Hit(doc_id=doc.doc_id, text=doc.text, score=score)
            for score, doc in scored[:k]
        ]

    def grep(
        self,
        pattern: str,
        *,
        k: int = 5,
        prefetch: int = 100,
    ) -> list[Bm25Hit]:
        if not pattern.strip():
            return []
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))
        hits = [
            Bm25Hit(doc_id=doc.doc_id, text=doc.text, score=1.0)
            for doc in self._documents
            if regex.search(doc.text)
        ]
        return hits[:k]

    def get_document(self, doc_id: str) -> str | None:
        norm = str(doc_id).split("_", 1)[0]
        for doc in self._documents:
            if doc.doc_id == norm:
                return doc.text
        return None
