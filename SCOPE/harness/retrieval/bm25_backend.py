"""BrowseComp+ official BM25 index backend (Pyserini / Lucene)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("harness.retrieval.bm25_backend")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX_ROOT = _REPO_ROOT / "external" / "BrowseComp-Plus" / "indexes" / "bm25"


@dataclass(frozen=True)
class Bm25Hit:
    doc_id: str
    text: str
    score: float


def resolve_bm25_index_path(path: str | Path | None = None) -> Path:
    """Resolve a Lucene BM25 index directory from env or default layout."""
    raw = path or os.environ.get("BROWSECOMP_BM25_INDEX_PATH") or str(_DEFAULT_INDEX_ROOT)
    candidate = Path(raw).expanduser().resolve()
    if _is_lucene_index_dir(candidate):
        return candidate

    if candidate.is_dir():
        for child in sorted(candidate.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir() and _is_lucene_index_dir(child):
                return child.resolve()

    raise FileNotFoundError(
        f"BM25 Lucene index not found at {candidate}. "
        "Run: bash scripts/setup_browsecomp_bm25_index.sh"
    )


def _is_lucene_index_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    return any(
        p.is_file()
        and p.name.startswith("segments_")
        and not p.name.endswith(".lock")
        for p in path.iterdir()
    )


class BrowseCompBm25Backend:
    """Thin wrapper around BrowseComp+ Pyserini BM25 index."""

    def __init__(self, index_path: str | Path) -> None:
        resolved = resolve_bm25_index_path(index_path)
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise ImportError(
                "pyserini is required for BM25 retrieval. "
                "Install with: pip install 'pyserini>=1.2.0'"
            ) from exc
        except Exception as exc:
            if "javac" in str(exc).lower() or "java" in str(exc).lower():
                raise RuntimeError(
                    "Java JDK is required for Pyserini/Lucene BM25 search. "
                    "Install OpenJDK 21, e.g.: conda install -c conda-forge openjdk=21"
                ) from exc
            raise

        self.index_path = resolved
        self._searcher = LuceneSearcher(str(resolved))
        logger.info("bm25_backend_ready", index_path=str(resolved))

    def search(
        self,
        query: str,
        *,
        k: int = 50,
        ignore_ids: set[str] | None = None,
    ) -> list[Bm25Hit]:
        if not query.strip():
            return []
        hits = self._searcher.search(query, max(k * 3, k))
        results: list[Bm25Hit] = []
        ignore = ignore_ids or set()
        for hit in hits:
            doc_id = str(hit.docid)
            if doc_id in ignore:
                continue
            text = self._extract_text(hit.lucene_document.get("raw"))
            results.append(Bm25Hit(doc_id=doc_id, text=text, score=float(hit.score)))
            if len(results) >= k:
                break
        return results

    def grep(
        self,
        pattern: str,
        *,
        k: int = 5,
        prefetch: int = 100,
    ) -> list[Bm25Hit]:
        """Regex grep via BM25 prefetch + in-memory filter (no Chroma regex API)."""
        if not pattern.strip():
            return []
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        pool = self.search(pattern, k=prefetch)
        matched = [hit for hit in pool if regex.search(hit.text)]
        if matched:
            return matched[:k]

        # Fallback: widen recall with a generic query.
        pool = self.search("document", k=prefetch)
        matched = [hit for hit in pool if regex.search(hit.text)]
        return matched[:k]

    def get_document(self, doc_id: str) -> str | None:
        norm = str(doc_id).split("_", 1)[0]
        doc = self._searcher.doc(norm)
        if doc is None:
            return None
        return self._extract_text(doc.raw())

    @staticmethod
    def _extract_text(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, dict):
            return str(raw.get("contents") or raw.get("text") or "")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            if isinstance(payload, dict):
                return str(payload.get("contents") or payload.get("text") or raw)
            return raw
        return str(raw)
