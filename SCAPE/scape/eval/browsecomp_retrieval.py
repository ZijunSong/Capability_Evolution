"""Optional BrowseComp-Plus retrieval backends for official four-cell eval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scape.eval.official_query_pool import default_bcp_root


@dataclass
class SearchHit:
    docid: str
    text: str
    score: float = 0.0


class RetrievalBackend:
    name = "none"

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        del query, k
        return []

    def get_doc(self, docid: str) -> str | None:
        del docid
        return None


class PyseriniBackend(RetrievalBackend):
    name = "pyserini_lucene"

    def __init__(self, index_dir: Path):
        from pyserini.search.lucene import LuceneSearcher

        self._searcher = LuceneSearcher(str(index_dir))

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        hits = []
        for hit in self._searcher.search(query, k):
            raw = ""
            try:
                raw = self._searcher.doc(hit.docid).raw()
            except Exception:
                raw = getattr(hit, "raw", "") or ""
            hits.append(SearchHit(str(hit.docid), str(raw), float(getattr(hit, "score", 0.0) or 0.0)))
        return hits

    def get_doc(self, docid: str) -> str | None:
        try:
            doc = self._searcher.doc(str(docid))
        except Exception:
            return None
        if doc is None:
            return None
        return str(doc.raw() or "")


class LocalJsonlBackend(RetrievalBackend):
    name = "local_corpus_token_overlap"

    def __init__(self, path: Path):
        import json
        import re

        self._docs: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                did = str(row.get("id") or row.get("docid") or row.get("source") or "")
                text = str(row.get("text") or row.get("contents") or row.get("content") or "")
                if did and text:
                    self._docs.append((did, text))
        self._tok = lambda s: {x for x in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        q = self._tok(query)
        scored: list[tuple[int, str, str]] = []
        for did, text in self._docs:
            overlap = len(q & self._tok(text))
            if overlap:
                scored.append((overlap, did, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [SearchHit(did, text, float(score)) for score, did, text in scored[:k]]

    def get_doc(self, docid: str) -> str | None:
        for did, text in self._docs:
            if did == str(docid):
                return text
        return None


def open_retrieval(bcp_root: Path | None = None) -> RetrievalBackend:
    root = bcp_root or default_bcp_root()
    if root is None:
        return RetrievalBackend()
    index = root / "indexes" / "bm25"
    if index.is_dir():
        try:
            return PyseriniBackend(index)
        except Exception:
            pass
    corpus = root / "data" / "browsecomp_plus_decrypted.jsonl"
    if corpus.is_file():
        return LocalJsonlBackend(corpus)
    return RetrievalBackend()


def norm_doc(docid: str) -> str:
    return str(docid).split("_", 1)[0]


def evidence_recall(retrieved: list[str], evidence_docids: list[str]) -> float:
    gold = {norm_doc(x) for x in evidence_docids}
    if not gold:
        return 0.0
    got = {norm_doc(x) for x in retrieved}
    return len(got & gold) / len(gold)


def hits_to_doc_store(hits: list[SearchHit]) -> dict[str, Any]:
    return {h.docid: {"id": h.docid, "text": h.text[:4000], "score": h.score} for h in hits}
