"""Harness-1 SEC RL queries + retrieval corpus.

Queries (scape+rl train pool)
    Default: ``/data/ppnm/harness-1-rl-data.tar.gz``
    Override: ``SCAPE_SEC_RL_DATA`` or ``--rl-data``
    Also accepts an extracted directory or a jsonl/parquet file.

Retrieval corpus (chunk texts, not the query pool)
    Default: ``/data/ppnm/harness-1-sec-corpus``
    Override: ``SCAPE_SEC_CORPUS_ROOT`` or ``--sec-corpus-root``
    Layout: ``corpora/sec/train/*.parquet`` with ``chunk_id`` / ``document_text``.
    Optional Lucene BM25: ``indexes/bm25``.

Eval stays on BrowseComp-Plus (full 830 for scape+rl). This module is the
train-time query + SEC retrieval contract only.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from scape.eval.browsecomp_retrieval import (
    PyseriniBackend,
    RetrievalBackend,
    SearchHit,
    _configure_java_runtime,
)
from scape.eval.official_query_pool import _as_id_list, _unique_ids

DEFAULT_SEC_RL_DATA = Path(
    os.environ.get("SCAPE_SEC_RL_DATA", "/data/ppnm/harness-1-rl-data.tar.gz")
)
DEFAULT_SEC_CORPUS_ROOT = Path(
    os.environ.get("SCAPE_SEC_CORPUS_ROOT", "/data/ppnm/harness-1-sec-corpus")
)
SEC_TRAIN_POOL_NAME = "harness-1-rl-data"
SEC_CORPUS_NAME = "harness-1-sec-corpus"

QUERY_MEMBERS = (
    "rl_queries_compact.jsonl",
    "rl_queries.jsonl",
    "rl_queries.json",
    "rl_queries.parquet",
    "queries.jsonl",
    "queries.json",
)
QUERY_DIR_CANDIDATES = QUERY_MEMBERS + (
    "queries/rl.jsonl",
    "queries/train.jsonl",
    "data/rl.jsonl",
)


def default_sec_rl_data() -> Path:
    env = os.environ.get("SCAPE_SEC_RL_DATA")
    if env:
        return Path(env)
    tar = Path("/data/ppnm/harness-1-rl-data.tar.gz")
    extracted = Path("/data/ppnm/harness-1-rl-data")
    if tar.is_file():
        return tar
    if extracted.exists():
        return extracted
    return tar


def default_sec_corpus_root() -> Path:
    return Path(os.environ.get("SCAPE_SEC_CORPUS_ROOT", str(DEFAULT_SEC_CORPUS_ROOT)))


def corpus_parquet_dir(root: Path | None = None) -> Path:
    return (root or default_sec_corpus_root()) / "corpora" / "sec" / "train"


def corpus_bm25_index(root: Path | None = None) -> Path:
    return (root or default_sec_corpus_root()) / "indexes" / "bm25"


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _chunk_ids_from_documents(raw: Any) -> tuple[list[str], list[str]]:
    docs = _parse_jsonish(raw)
    evidence: list[str] = []
    gold: list[str] = []
    if isinstance(docs, dict):
        docs = docs.get("document_ids") or docs.get("chunk_ids") or [docs]
    if not isinstance(docs, list):
        return [], []
    for item in docs:
        if isinstance(item, str):
            evidence.append(item)
            gold.append(item)
            continue
        if not isinstance(item, dict):
            continue
        cids = [str(x) for x in (item.get("chunk_ids") or item.get("docids") or []) if str(x)]
        if not cids and item.get("chunk_id"):
            cids = [str(item["chunk_id"])]
        evidence.extend(cids)
        if item.get("is_final_answer") is False:
            continue
        gold.extend(cids)
    if not gold:
        gold = list(evidence)
    return _unique_ids(evidence), _unique_ids(gold)


def normalize_sec_rl_record(raw: Any) -> dict[str, Any] | None:
    """Map a Harness-1 RL query row onto SCAPE's query_id / gold_docids contract."""
    if isinstance(raw, str):
        return {"query_id": raw, "query": raw, "gold_docids": [], "evidence_docids": []}
    if not isinstance(raw, dict):
        return None
    payload = _parse_jsonish(raw.get("payload_json") or {})
    if not isinstance(payload, dict):
        payload = {}
    qid = str(raw.get("query_id") or payload.get("query_id") or raw.get("id") or "")
    if not qid:
        return None
    query = str(
        raw.get("query")
        or raw.get("question")
        or payload.get("query")
        or qid
    )
    evidence, gold = _chunk_ids_from_documents(
        raw.get("document_ids")
        or raw.get("document_ids_json")
        or payload.get("document_ids")
        or raw.get("gold_document_ids")
        or raw.get("gold_docids")
        or []
    )
    if not gold:
        gold = _as_id_list(raw.get("gold_docids") or raw.get("gold_document_ids") or [])
    if not evidence:
        evidence = list(gold)
    rec = {
        "query_id": qid,
        "query": query,
        "answer": str(raw.get("answer") or payload.get("answer") or ""),
        "evidence_docids": evidence,
        "gold_docids": gold,
        "official_split": "train",
        "train_pool": SEC_TRAIN_POOL_NAME,
        "dataset_name": str(raw.get("dataset_name") or payload.get("dataset_name") or "sec"),
        "stage": str(raw.get("stage") or "rl"),
    }
    return rec


def _read_json_bytes(data: bytes, *, suffix: str) -> list[Any]:
    text = data.decode("utf-8")
    if suffix.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("queries") or payload.get("records") or payload.get("data") or []
        if rows:
            return list(rows)
        if payload.get("query_ids"):
            return [{"query_id": str(q), "query": str(q)} for q in payload["query_ids"]]
    return []


def _read_parquet_bytes(data: bytes) -> list[Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(data))
    return list(table.to_pylist())


def _read_parquet_path(path: Path) -> list[Any]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_table(path).to_pylist())
    except ImportError:
        pass
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    except Exception as exc:
        raise FileNotFoundError(f"cannot read parquet query file {path}: {exc}") from exc


def _iter_dir_query_files(root: Path) -> list[Path]:
    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        named = payload.get("queries_file") or payload.get("rl_queries_file")
        if named:
            path = Path(named) if Path(named).is_absolute() else root / str(named)
            if path.is_file():
                return [path]
    found: list[Path] = []
    for rel in QUERY_DIR_CANDIDATES:
        path = root / rel
        if path.is_file():
            found.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _load_rows_from_path(path: Path) -> tuple[list[Any], str]:
    if path.suffix == ".parquet":
        return _read_parquet_path(path), str(path)
    if path.suffix in {".jsonl", ".json"}:
        return _read_json_bytes(path.read_bytes(), suffix=path.suffix), str(path)
    raise FileNotFoundError(f"unsupported query file {path}")


def _load_rows_from_tar(archive: Path) -> tuple[list[Any], str]:
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        by_name = {Path(m.name).name: m for m in members}
        for name in QUERY_MEMBERS:
            member = by_name.get(name)
            if member is None:
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            data = handle.read()
            suffix = Path(member.name).suffix
            rows = _read_parquet_bytes(data) if suffix == ".parquet" else _read_json_bytes(data, suffix=suffix)
            return rows, f"{archive}::{member.name}"
    raise FileNotFoundError(
        f"{archive} has no RL query file ({', '.join(QUERY_MEMBERS[:3])})"
    )


def resolve_sec_rl_query_source(source: Path | None = None) -> Path:
    path = Path(source) if source is not None else default_sec_rl_data()
    if path.is_file() or path.is_dir():
        return path
    sibling = path.with_suffix("") if path.name.endswith(".tar.gz") else path
    if path.name.endswith(".tar.gz") and Path(str(path)[: -len(".tar.gz")]).is_dir():
        return Path(str(path)[: -len(".tar.gz")])
    if sibling.is_dir():
        return sibling
    return path


def load_sec_rl_queries(
    source: Path | None = None,
    *,
    n_queries: int | None = None,
    query_file: Path | None = None,
    corpus_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load Harness-1 SEC RL training queries (3453 official records)."""
    if query_file is not None:
        path = Path(query_file)
        if not path.is_file():
            raise FileNotFoundError(f"SEC RL query file not found: {path}")
        raw, loc = _load_rows_from_path(path)
        origin = path
    else:
        origin = resolve_sec_rl_query_source(source)
        if origin.is_dir():
            files = _iter_dir_query_files(origin)
            if not files:
                raise FileNotFoundError(
                    f"SEC RL queries not found under {origin}. "
                    "Upload harness-1-rl-data.tar.gz (or extract it) and pass --rl-data."
                )
            raw, loc = _load_rows_from_path(files[0])
        elif origin.is_file() and origin.name.endswith(".tar.gz"):
            raw, loc = _load_rows_from_tar(origin)
        elif origin.is_file():
            raw, loc = _load_rows_from_path(origin)
        else:
            raise FileNotFoundError(
                f"SEC RL queries not found at {origin}. "
                "Upload /data/ppnm/harness-1-rl-data.tar.gz or pass --rl-data."
            )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        rec = normalize_sec_rl_record(item)
        if rec is None or rec["query_id"] in seen:
            continue
        seen.add(rec["query_id"])
        rows.append(rec)
    if not rows:
        raise RuntimeError(f"SEC RL query source {loc} produced zero records")
    used = list(rows)
    if n_queries not in {None, 0} and int(n_queries) < len(used):
        used = used[: int(n_queries)]
    corpus = Path(corpus_root) if corpus_root is not None else default_sec_corpus_root()
    parquet = corpus_parquet_dir(corpus)
    meta = {
        "path": loc,
        "source": str(origin),
        "root": str(origin),
        "pool_contract": SEC_TRAIN_POOL_NAME,
        "query_count": len(used),
        "query_count_available": len(rows),
        "using_full_train_split": n_queries in {None, 0} or int(n_queries) >= len(rows),
        "corpus_root": str(corpus),
        "corpus_name": SEC_CORPUS_NAME,
        "corpus_parquet_dir": str(parquet),
        "corpus_parquet_present": parquet.is_dir()
        and any(parquet.glob("*.parquet")),
        "corpus_bm25_index": str(corpus_bm25_index(corpus)),
        "corpus_bm25_present": corpus_bm25_index(corpus).is_dir(),
    }
    return used, meta


def lookup_chunk_texts(
    chunk_ids: Sequence[str],
    parquet_dir: Path | None = None,
) -> dict[str, str]:
    """Look up SEC chunk texts by id. No-op if the parquet corpus is absent."""
    wanted = {str(x) for x in chunk_ids if str(x)}
    root = Path(parquet_dir) if parquet_dir is not None else corpus_parquet_dir()
    if not wanted or not root.is_dir():
        return {}
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        return {}
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ImportError:
        return {}
    dataset = ds.dataset([str(p) for p in shards], format="parquet")
    cols = set(dataset.schema.names)
    id_col = "chunk_id" if "chunk_id" in cols else ("id" if "id" in cols else None)
    text_col = "document_text" if "document_text" in cols else ("text" if "text" in cols else None)
    if not id_col or not text_col:
        return {}
    table = dataset.to_table(columns=[id_col, text_col], filter=pc.field(id_col).isin(list(wanted)))
    out: dict[str, str] = {}
    for cid, text in zip(table[id_col].to_pylist(), table[text_col].to_pylist()):
        key = str(cid)
        if key in wanted and key not in out:
            out[key] = str(text or "")
    return out


def attach_sec_doc_stores(
    rows: Sequence[dict[str, Any]],
    *,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    """Fill ``seed_doc_store`` from gold/evidence chunk ids when parquet is present."""
    parquet = corpus_parquet_dir(corpus_root)
    ids: list[str] = []
    for row in rows:
        ids.extend(row.get("gold_docids") or [])
        ids.extend(row.get("evidence_docids") or [])
    texts = lookup_chunk_texts(_unique_ids(ids), parquet)
    n_seeded = 0
    for row in rows:
        store: dict[str, Any] = {}
        for did in list(row.get("gold_docids") or []) + list(row.get("evidence_docids") or []):
            text = texts.get(str(did))
            if not text:
                continue
            store[str(did)] = {"id": str(did), "text": text[:4000], "score": 1.0}
        if store:
            row["seed_doc_store"] = store
            n_seeded += 1
    return {
        "n_chunk_texts": len(texts),
        "n_rows_seeded": n_seeded,
        "corpus_parquet_dir": str(parquet),
        "corpus_parquet_present": bool(texts) or (parquet.is_dir() and any(parquet.glob("*.parquet"))),
    }


class SecCorpusBackend(RetrievalBackend):
    """SEC retrieval: Lucene BM25 if indexed, else token-overlap over cached chunks."""

    name = "sec_parquet"

    def __init__(self, root: Path | None = None, *, texts: dict[str, str] | None = None):
        self.root = Path(root) if root is not None else default_sec_corpus_root()
        self.parquet_dir = corpus_parquet_dir(self.root)
        self._texts: dict[str, str] = dict(texts or {})
        self._lucene: PyseriniBackend | None = None
        index = corpus_bm25_index(self.root)
        if index.is_dir():
            try:
                _configure_java_runtime()
                self._lucene = PyseriniBackend(index)
                self.name = "sec_pyserini"
            except Exception:
                self._lucene = None
        if self._lucene is None and not self.parquet_dir.is_dir() and not self._texts:
            self.name = "none"

    def prefetch(self, chunk_ids: Iterable[str]) -> dict[str, str]:
        missing = [str(x) for x in chunk_ids if str(x) and str(x) not in self._texts]
        if missing:
            self._texts.update(lookup_chunk_texts(missing, self.parquet_dir))
        return {cid: self._texts[cid] for cid in chunk_ids if cid in self._texts}

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if self._lucene is not None:
            return self._lucene.search(query, k)
        if self.name == "none" or not self._texts:
            return []
        import re

        q = {x for x in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
        scored: list[tuple[int, str, str]] = []
        for did, text in self._texts.items():
            toks = {x for x in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}
            overlap = len(q & toks)
            if overlap:
                scored.append((overlap, did, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [SearchHit(did, text, float(score)) for score, did, text in scored[:k]]

    def get_doc(self, docid: str) -> str | None:
        key = str(docid)
        if key in self._texts:
            return self._texts[key]
        if self._lucene is not None:
            return self._lucene.get_doc(key)
        found = lookup_chunk_texts([key], self.parquet_dir)
        if key in found:
            self._texts[key] = found[key]
            return found[key]
        return None


def open_sec_retrieval(root: Path | None = None, *, texts: dict[str, str] | None = None) -> RetrievalBackend:
    backend = SecCorpusBackend(root, texts=texts)
    return backend
