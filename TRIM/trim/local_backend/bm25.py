"""Local Lucene/Pyserini BM25. Token-overlap JSONL is not a fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trim.eval.browsecomp_retrieval import PyseriniBackend, lucene_stored_text
from trim.local_backend.corpus_store import Chunk, LocalCorpusStore


@dataclass(frozen=True)
class RankHit:
    chunk_id: str
    official_id: str
    text: str
    score: float


class LocalBm25Backend:
    name = "local_bm25"

    def __init__(
        self,
        *,
        index_path: Path,
        store: LocalCorpusStore,
        k1: float = 0.9,
        b: float = 0.4,
    ) -> None:
        if not Path(index_path).exists():
            raise RuntimeError(f"local_bm25 index_path does not exist: {index_path}")
        self.index_path = Path(index_path)
        self.store = store
        self.k1 = float(k1)
        self.b = float(b)
        self._lucene = PyseriniBackend(self.index_path)
        try:
            self._lucene.configure_bm25(self.k1, self.b)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to configure BM25 parameters k1={self.k1} b={self.b}: {exc}"
            ) from exc
        self.bm25_configure_ok = True
        self.index_version = f"lucene:{self.index_path.resolve()}#docs={self._lucene.num_docs()}"
        self.store.manifest.index_path = str(self.index_path)
        self.store.manifest.index_version = self.index_version

    def num_docs(self) -> int:
        return int(self._lucene.num_docs())

    def search(
        self,
        query: str,
        *,
        k: int,
        ignore_ids: list[str] | None = None,
    ) -> list[RankHit]:
        ignore = {str(x) for x in (ignore_ids or [])}
        fetch = max(int(k) * 3, int(k))
        raw = list(self._lucene.search(query, fetch) or [])
        hits: list[RankHit] = []
        for item in raw:
            official = self.store.id_map.official_of(item.docid)
            internal_root = self.store.id_map.internal_of(official)
            if item.docid in ignore or official in ignore or internal_root in ignore:
                continue
            chunk = self._resolve_chunk(item.docid, official, item.text)
            if chunk is None:
                continue
            if chunk.chunk_id in ignore:
                continue
            text = chunk.text if chunk.text else item.text
            hits.append(RankHit(chunk_id=chunk.chunk_id, official_id=official, text=text, score=float(item.score)))
            if len(hits) >= int(k):
                break
        return hits

    def _resolve_chunk(self, hit_id: str, official: str, lucene_text: str) -> Chunk | None:
        direct = self.store.get_chunk(hit_id)
        if direct is not None:
            return direct
        chunks = self.store.chunks_for(official)
        if len(chunks) == 1:
            return chunks[0]
        if chunks:
            return chunks[0]
        if lucene_text:
            cid = self.store.id_map.chunk_id(official, 0)
            return Chunk(chunk_id=cid, doc_id=official, ordinal=0, text=lucene_text)
        return None

    def lucene_raw(self, docid: str) -> str:
        text = self._lucene.get_doc(docid) or ""
        return lucene_stored_text(text)
