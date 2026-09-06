"""Local-retrieval transfer benchmarks for TRIM eval.

Harness-1 Table 2 uses a mix of Chroma corpora and live web. TRIM eval stays
offline: each benchmark is a query jsonl plus a Pyserini/JSONL corpus whose
document IDs match ``gold_docids`` / ``evidence_docids``.

Built by ``scripts/build_transfer_local_corpus.py`` into
``TRIM/manifests/transfer_local/{name}/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from trim.eval.browsecomp_retrieval import (
    LocalJsonlBackend,
    PyseriniBackend,
    RetrievalBackend,
    _configure_java_runtime,
    assert_retrieval_ready,
    open_retrieval,
)
from trim.eval.official_query_pool import (
    SCORE_SPLIT_166,
    SCORE_SPLIT_830,
    canonical_score_split,
    is_full_score_split,
    load_bcplus_830_full,
    load_bcplus_830_split,
    load_query_manifest,
)

REPO = Path(__file__).resolve().parents[2]
CHUNK_SEP = "::c"

TRANSFER_BENCHMARKS = ("longsealqa", "frames", "hotpotqa")
OPTIONAL_PRIVATE_BENCHMARKS = ("web", "patents")
LOCAL_EVAL_BENCHMARKS = TRANSFER_BENCHMARKS + OPTIONAL_PRIVATE_BENCHMARKS

BCPLUS_BENCHMARKS = frozenset({"BC+", "bcplus_test_166", "bcplus_full", "bcplus_830"})

_TRANSFER_ALIASES = {
    "longseal": "longsealqa",
    "longsealqa": "longsealqa",
    "longseal-qa": "longsealqa",
    "long_seal": "longsealqa",
    "long_seal_qa": "longsealqa",
    "frames": "frames",
    "frames-benchmark": "frames",
    "google_frames": "frames",
    "hotpotqa": "hotpotqa",
    "hotpot": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "hotpotqa_subset": "hotpotqa",
    "web": "web",
    "web_synthetic": "web",
    "web_1_17": "web",
    "patents": "patents",
    "uspto": "patents",
    "patent": "patents",
}


def canonical_transfer_benchmark(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "")
    return _TRANSFER_ALIASES.get(key)


def is_bcplus_benchmark(benchmark: str) -> bool:
    key = str(benchmark or "").strip()
    if key in BCPLUS_BENCHMARKS:
        return True
    split = canonical_score_split(key)
    return split in {SCORE_SPLIT_166, SCORE_SPLIT_830}


def is_local_eval_benchmark(benchmark: str) -> bool:
    return canonical_transfer_benchmark(benchmark) in LOCAL_EVAL_BENCHMARKS


def default_transfer_root() -> Path:
    env = os.environ.get("TRIM_TRANSFER_CORPUS_ROOT")
    if env:
        return Path(env)
    return REPO / "manifests" / "transfer_local"


def transfer_dir(name: str, *, root: Path | None = None) -> Path:
    canon = canonical_transfer_benchmark(name)
    if canon is None:
        raise ValueError(f"not a transfer benchmark: {name}")
    return (root or default_transfer_root()) / canon


def parent_chunk_id(docid: str) -> str:
    text = str(docid or "")
    if CHUNK_SEP in text:
        return text.rsplit(CHUNK_SEP, 1)[0]
    return text


def chunk_doc_id(docid: str, index: int) -> str:
    return f"{docid}{CHUNK_SEP}{int(index)}"


def normalize_url_id(docid: str) -> str:
    text = parent_chunk_id(docid).strip()
    if "://" not in text:
        return text
    parsed = urlsplit(text)
    path = unquote(parsed.path or "")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path.rstrip("/"), parsed.query, ""))


def wiki_title_key(value: str) -> str:
    text = parent_chunk_id(value).strip()
    if not text:
        return ""
    if "://" in text or text.startswith("/wiki/"):
        parsed = urlsplit(text if "://" in text else f"https://en.wikipedia.org{text}")
        path = unquote(parsed.path or "")
        marker = "/wiki/"
        if marker in path:
            path = path.split(marker, 1)[1]
        text = path
    text = unquote(text).replace(" ", "_")
    if text.startswith("en.wikipedia.org/wiki/"):
        text = text.split("/wiki/", 1)[1]
    return text.strip("/").replace("_", " ").strip()


def recall_keys(docid: str) -> set[str]:
    raw = str(docid or "")
    keys = {raw, parent_chunk_id(raw), normalize_url_id(raw)}
    title = wiki_title_key(raw)
    if title:
        keys.add(title)
        keys.add(title.replace(" ", "_"))
    return {k for k in keys if k}


class TransferRetrievalBackend(RetrievalBackend):
    """Lucene/JSONL backend that matches gold IDs at document (not chunk) granularity."""

    name = "transfer_lucene"
    id_style = "chunk_parent"

    def __init__(self, inner: RetrievalBackend):
        self._inner = inner
        self.name = f"transfer_{inner.name}"

    def search(self, query: str, k: int = 5):
        return self._inner.search(query, k)

    def get_doc(self, docid: str) -> str | None:
        return self._inner.get_doc(docid)

    def num_docs(self) -> int:
        n_docs = getattr(self._inner, "num_docs", None)
        if callable(n_docs):
            return int(n_docs())
        return int(n_docs or 0)

    def normalize_id(self, docid: str) -> str:
        parent = parent_chunk_id(docid)
        if "wikipedia.org" in parent.lower() or "/wiki/" in parent:
            return wiki_title_key(parent)
        return normalize_url_id(parent)


def _is_lucene_index(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.name.startswith("segments_") and p.is_file() for p in path.iterdir())


def open_transfer_retrieval(name: str, *, root: Path | None = None, formal: bool = True) -> RetrievalBackend:
    canon = canonical_transfer_benchmark(name)
    if canon is None:
        raise ValueError(f"not a transfer benchmark: {name}")
    base = (root or default_transfer_root()) / canon
    index = base / "indexes" / "bm25"
    corpus = base / "corpus.jsonl"
    if _is_lucene_index(index):
        try:
            _configure_java_runtime()
            inner = PyseriniBackend(index)
            backend = TransferRetrievalBackend(inner)
            assert_retrieval_ready(inner, formal=formal)
            return backend
        except Exception:
            if formal and not corpus.is_file():
                raise
    if corpus.is_file():
        inner = LocalJsonlBackend(corpus)
        backend = TransferRetrievalBackend(inner)
        backend.name = "transfer_local_jsonl"
        if formal:
            probe = inner.search("the", 1) or inner.search("history", 1)
            if not probe:
                # corpus may not contain those probe words; accept nonempty file
                if corpus.stat().st_size <= 2:
                    raise RuntimeError(f"empty transfer corpus: {corpus}")
        return backend
    if formal:
        raise RuntimeError(
            f"local corpus for --benchmark {canon} is missing under {base}. "
            f"Build it with: PYTHONPATH=TRIM python TRIM/scripts/build_transfer_local_corpus.py --benchmark {canon}"
        )
    return RetrievalBackend()


def open_eval_retrieval(benchmark: str, *, formal: bool = True) -> RetrievalBackend:
    if is_bcplus_benchmark(benchmark) or canonical_transfer_benchmark(benchmark) is None:
        return open_retrieval(formal=formal)
    return open_transfer_retrieval(benchmark, formal=formal)


def _tag_test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("official_split", "test")
        rec.setdefault("gold_docids", list(rec.get("gold_docids") or []))
        rec.setdefault("evidence_docids", list(rec.get("evidence_docids") or rec["gold_docids"]))
        out.append(rec)
    return out


def load_transfer_queries(name: str, *, root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canon = canonical_transfer_benchmark(name)
    if canon is None:
        raise ValueError(f"not a transfer benchmark: {name}")
    base = (root or default_transfer_root()) / canon
    path = base / "queries.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"query pool missing: {path}. "
            f"Build it with: PYTHONPATH=TRIM python TRIM/scripts/build_transfer_local_corpus.py --benchmark {canon}"
        )
    rows = _tag_test(load_query_manifest(path))
    build = base / "BUILD.json"
    extra: dict[str, Any] = {}
    if build.is_file():
        extra = json.loads(build.read_text(encoding="utf-8"))
    meta = {
        "path": str(path),
        "pool_contract": f"transfer_{canon}",
        "query_count": len(rows),
        "eval_count": len(rows),
        "score_split": canon,
        "primary_eval": canon,
        "official_test_count": len(rows),
        "official_test_expected": len(rows),
        "build": extra,
    }
    return rows, meta


def load_eval_benchmark(
    benchmark: str,
    *,
    score_split: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the eval query pool implied by ``--benchmark`` / ``--score-split``."""
    transfer = canonical_transfer_benchmark(benchmark)
    if transfer:
        return load_transfer_queries(transfer)
    split = canonical_score_split(score_split) if score_split else None
    if split == SCORE_SPLIT_166 or str(benchmark).strip() == "bcplus_test_166":
        _train, rows, meta = load_bcplus_830_split()
        meta = dict(meta)
        meta.update(
            {
                "query_count": len(rows),
                "eval_count": len(rows),
                "primary_eval": SCORE_SPLIT_166,
                "score_split": SCORE_SPLIT_166,
            }
        )
        return rows, meta
    rows, meta = load_bcplus_830_full()
    return rows, meta


def score_split_for_eval_benchmark(benchmark: str) -> str | None:
    transfer = canonical_transfer_benchmark(benchmark)
    if transfer:
        return transfer
    key = str(benchmark or "").strip()
    if key == "bcplus_test_166":
        return SCORE_SPLIT_166
    if key in {"bcplus_full", "bcplus_830"}:
        return SCORE_SPLIT_830
    return None
