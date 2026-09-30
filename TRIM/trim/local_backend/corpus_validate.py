"""Validate local corpus coverage against a Lucene index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from trim.local_backend.corpus_store import LocalCorpusStore


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
    probes = list(probe_docids or ("59931", "69324", "44797"))
    missing_probes = [did for did in probes if store.get_document(str(did)) is None]
    # Transfer benchmarks (longsealqa/frames/hotpotqa) use URL-style doc ids
    # (e.g. "https://en.wikipedia.org/...::c0"), not BrowseComp-Plus numeric
    # ids. When the BC+ default probes all miss, fall back to probing the first
    # few actual store ids — this verifies read_document works for this corpus
    # without assuming the BC+ id scheme. BC+ corpora still hit the default
    # probes and take the fast path.
    if missing_probes and not probe_docids and n_store > 0:
        sample_ids = list(store_ids)[: min(3, n_store)]
        missing_sample = [did for did in sample_ids if store.get_document(str(did)) is None]
        if not missing_sample:
            probes = sample_ids
            missing_probes = []
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
            f"Lucene index reports {n_index}. Rebuild with "
            "TRIM/scripts/build_browsecomp_corpus_from_index.py --mode full."
        )
    if missing_probes:
        raise RuntimeError(
            f"local_bm25 corpus missing probe documents: {missing_probes}. "
            "Search hits on these ids would fail read_document."
        )
    return report
