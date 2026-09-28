"""Validate local corpus coverage against a Lucene index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from trim.local_backend.corpus_store import LocalCorpusStore

BCPLUS_PROBE_DOCIDS = ("59931", "69324", "44797")


def _default_probe_docids(store: LocalCorpusStore) -> list[str]:
    if all(store.get_document(did) is not None for did in BCPLUS_PROBE_DOCIDS):
        return list(BCPLUS_PROBE_DOCIDS)
    return [str(did) for did in list(store.documents.keys())[:3]]


def validate_corpus_against_index(
    *,
    store: LocalCorpusStore,
    index_num_docs: int,
    index_path: Path | None = None,
    probe_docids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Ensure docstore size matches the index and probe documents are readable."""
    store_ids = set(store.documents.keys())
    n_store = len(store_ids)
    n_index = int(index_num_docs)
    probes = list(probe_docids) if probe_docids is not None else _default_probe_docids(store)
    missing_probes = [did for did in probes if store.get_document(str(did)) is None]
    report: dict[str, Any] = {
        "ok": n_store == n_index and not missing_probes,
        "index_path": str(index_path or store.manifest.index_path or ""),
        "corpus_path": str(store.manifest.corpus_path or ""),
        "index_num_docs": n_index,
        "corpus_num_docs": n_store,
        "doc_count_delta": n_store - n_index,
        "probe_docids": probes,
        "missing_probe_docids": missing_probes,
    }
    if n_store != n_index:
        raise RuntimeError(
            f"local_bm25 corpus/index mismatch: corpus has {n_store} documents but "
            f"Lucene index reports {n_index}. Rebuild the local corpus/index for this "
            "--benchmark (BC+: TRIM/scripts/build_browsecomp_corpus_from_index.py --mode full; "
            "transfer: TRIM/scripts/build_transfer_local_corpus.py)."
        )
    if missing_probes:
        raise RuntimeError(
            f"local_bm25 corpus missing probe documents: {missing_probes}. "
            "Search hits on these ids would fail read_document."
        )
    return report
