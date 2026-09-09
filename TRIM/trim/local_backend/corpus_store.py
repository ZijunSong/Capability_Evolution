"""Immutable local documents and chunks. Display clips never write back."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from trim.local_backend.id_map import IdMap

CHUNK_STRATEGY_DOCUMENT = "document_as_single_chunk"
CHUNK_STRATEGY_PROVIDED = "provided_chunks"


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str = ""
    summary: str = ""

    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    start: int | None = None
    end: int | None = None

    def parent_id(self) -> str:
        return self.doc_id


@dataclass
class CorpusManifest:
    corpus_version: str = "unknown"
    n_documents: int = 0
    n_chunks: int = 0
    chunk_strategy: str = CHUNK_STRATEGY_DOCUMENT
    index_version: str = "unknown"
    index_path: str = ""
    corpus_path: str = ""
    docstore_path: str = ""
    id_map_path: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalCorpusStore:
    """Full-text store. Search snippets must not mutate these records."""

    def __init__(
        self,
        *,
        documents: Mapping[str, Document],
        chunks: Mapping[str, Chunk],
        id_map: IdMap,
        manifest: CorpusManifest,
    ) -> None:
        self.documents = dict(documents)
        self.chunks = dict(chunks)
        self.id_map = id_map
        self.manifest = manifest
        self._chunks_by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.chunks.values():
            self._chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
        for doc_id, items in self._chunks_by_doc.items():
            items.sort(key=lambda c: c.ordinal)
        self._checksums = {did: doc.checksum() for did, doc in self.documents.items()}

    def checksum_of(self, official_id: str) -> str:
        return self._checksums.get(str(official_id), "")

    def get_document(self, official_or_internal: str) -> Document | None:
        official = self.id_map.official_of(official_or_internal)
        return self.documents.get(official)

    def chunks_for(self, official_or_internal: str) -> list[Chunk]:
        official = self.id_map.official_of(official_or_internal)
        return list(self._chunks_by_doc.get(official) or [])

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        if chunk_id in self.chunks:
            return self.chunks[chunk_id]
        official = self.id_map.official_of(chunk_id)
        for chunk in self._chunks_by_doc.get(official) or []:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def neighbors(self, chunk_id: str) -> list[Chunk]:
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return []
        ordered = self.chunks_for(chunk.doc_id)
        idx = next((i for i, item in enumerate(ordered) if item.chunk_id == chunk.chunk_id), None)
        if idx is None:
            return []
        out: list[Chunk] = []
        if idx > 0:
            out.append(ordered[idx - 1])
        if idx + 1 < len(ordered):
            out.append(ordered[idx + 1])
        return out

    def iter_chunks(self) -> Iterator[Chunk]:
        for items in self._chunks_by_doc.values():
            yield from items

    def assert_immutable(self, official_id: str, expected: str | None = None) -> None:
        current = self.checksum_of(official_id)
        if expected is not None and current != expected:
            raise RuntimeError(f"docstore mutated for {official_id}")

    @classmethod
    def from_jsonl(cls, path: Path, *, id_map: IdMap | None = None, manifest: CorpusManifest | None = None) -> "LocalCorpusStore":
        id_map = id_map or IdMap()
        documents: dict[str, Document] = {}
        chunks: dict[str, Chunk] = {}
        strategy = CHUNK_STRATEGY_DOCUMENT
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                official = str(row.get("id") or row.get("docid") or row.get("doc_id") or row.get("source") or "")
                text = str(row.get("text") or row.get("contents") or row.get("content") or "")
                if not official:
                    continue
                id_map.register_official(official)
                documents[official] = Document(
                    doc_id=official,
                    text=text,
                    source=str(row.get("source") or official),
                    summary=str(row.get("summary") or "")[:500],
                )
                provided = row.get("chunks")
                if isinstance(provided, list) and provided:
                    strategy = CHUNK_STRATEGY_PROVIDED
                    for i, item in enumerate(provided):
                        if isinstance(item, dict):
                            ctext = str(item.get("text") or item.get("contents") or "")
                            ordinal = int(item.get("ordinal") if item.get("ordinal") is not None else i)
                        else:
                            ctext = str(item)
                            ordinal = i
                        cid = id_map.chunk_id(official, ordinal)
                        chunks[cid] = Chunk(chunk_id=cid, doc_id=official, ordinal=ordinal, text=ctext)
                else:
                    cid = id_map.chunk_id(official, 0)
                    chunks[cid] = Chunk(chunk_id=cid, doc_id=official, ordinal=0, text=text, start=0, end=len(text))
        man = manifest or CorpusManifest(
            corpus_path=str(path),
            n_documents=len(documents),
            n_chunks=len(chunks),
            chunk_strategy=strategy,
        )
        man.n_documents = len(documents)
        man.n_chunks = len(chunks)
        man.chunk_strategy = strategy
        man.corpus_path = str(path)
        return cls(documents=documents, chunks=chunks, id_map=id_map, manifest=man)

    @classmethod
    def from_sqlite(cls, path: Path, *, id_map: IdMap | None = None, manifest: CorpusManifest | None = None) -> "LocalCorpusStore":
        id_map = id_map or IdMap()
        documents: dict[str, Document] = {}
        chunks: dict[str, Chunk] = {}
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute("SELECT doc_id, text, source, summary FROM documents"):
                official = str(row["doc_id"])
                id_map.register_official(official)
                documents[official] = Document(
                    doc_id=official,
                    text=str(row["text"] or ""),
                    source=str(row["source"] or official),
                    summary=str(row["summary"] or ""),
                )
            chunk_rows = list(conn.execute("SELECT chunk_id, doc_id, ordinal, text, start_char, end_char FROM chunks"))
        finally:
            conn.close()
        strategy = CHUNK_STRATEGY_PROVIDED if chunk_rows else CHUNK_STRATEGY_DOCUMENT
        if chunk_rows:
            for row in chunk_rows:
                official = str(row["doc_id"])
                ordinal = int(row["ordinal"] or 0)
                cid = str(row["chunk_id"] or id_map.chunk_id(official, ordinal))
                chunks[cid] = Chunk(
                    chunk_id=cid,
                    doc_id=official,
                    ordinal=ordinal,
                    text=str(row["text"] or ""),
                    start=row["start_char"],
                    end=row["end_char"],
                )
        else:
            for official, doc in documents.items():
                cid = id_map.chunk_id(official, 0)
                chunks[cid] = Chunk(chunk_id=cid, doc_id=official, ordinal=0, text=doc.text, start=0, end=len(doc.text))
        man = manifest or CorpusManifest(docstore_path=str(path), n_documents=len(documents), n_chunks=len(chunks), chunk_strategy=strategy)
        man.n_documents = len(documents)
        man.n_chunks = len(chunks)
        man.docstore_path = str(path)
        man.chunk_strategy = strategy
        return cls(documents=documents, chunks=chunks, id_map=id_map, manifest=man)

    @classmethod
    def from_memory(cls, rows: Iterable[Mapping[str, Any]], *, id_map: IdMap | None = None) -> "LocalCorpusStore":
        id_map = id_map or IdMap()
        documents: dict[str, Document] = {}
        chunks: dict[str, Chunk] = {}
        for row in rows:
            official = str(row.get("id") or row.get("doc_id") or "")
            text = str(row.get("text") or "")
            if not official:
                continue
            id_map.register_official(official)
            documents[official] = Document(doc_id=official, text=text, source=official)
            cid = id_map.chunk_id(official, 0)
            chunks[cid] = Chunk(chunk_id=cid, doc_id=official, ordinal=0, text=text, start=0, end=len(text))
        return cls(
            documents=documents,
            chunks=chunks,
            id_map=id_map,
            manifest=CorpusManifest(
                n_documents=len(documents),
                n_chunks=len(chunks),
                chunk_strategy=CHUNK_STRATEGY_DOCUMENT,
                notes=["in_memory_test_store"],
            ),
        )
