#!/usr/bin/env python3
"""Build the official Harness-G corpus graph from BrowseComp-Plus.

Reads the full BM25-aligned JSONL (same docid namespace as Lucene) and writes
a ``scope=corpus`` graph index. This is the artifact official Harness-G eval
must load via ``--graph-index-path`` / ``GRAPH_INDEX_PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.harness_g_contract import validate_loaded_graph
from trim.eval.harness_g_graph import HarnessGGraphIndex, save_graph_index
from trim.eval.official_query_pool import default_bcp_root


def default_corpus_jsonl() -> Path | None:
    root = default_bcp_root()
    if root is None:
        return None
    cand = root / "data" / "browsecomp_plus_corpus_full.jsonl"
    return cand if cand.is_file() else None


def default_out_path() -> Path:
    root = default_bcp_root()
    if root is None:
        return _TRIM / "manifests" / "harness_g" / "bcplus_corpus_graph.pkl"
    return root / "indexes" / "harness_g_corpus_graph.pkl"


def iter_corpus(path: Path, *, limit: int | None = None):
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            did = str(row.get("id") or row.get("docid") or row.get("doc_id") or "").strip()
            text = str(row.get("text") or row.get("contents") or row.get("document_text") or "")
            if not did or not text.strip():
                continue
            yield did, {"id": did, "text": text, "parent_docid": did}
            n += 1
            if limit is not None and n >= int(limit):
                return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build official Harness-G BC+ corpus graph")
    parser.add_argument("--corpus-jsonl", type=Path, default=default_corpus_jsonl())
    parser.add_argument("--out", type=Path, default=default_out_path())
    parser.add_argument("--scope", default="corpus")
    parser.add_argument("--source-corpus", default="browsecomp_plus")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args(argv)

    if args.corpus_jsonl is None or not Path(args.corpus_jsonl).is_file():
        raise SystemExit(
            "BrowseComp-Plus full corpus JSONL not found. "
            "Build it with scripts/build_browsecomp_corpus_from_index.py --mode full"
        )

    graph = HarnessGGraphIndex(scope=str(args.scope))
    graph.source_corpus = str(args.source_corpus)
    t0 = time.perf_counter()
    batch: dict[str, dict] = {}
    n_seen = 0
    n_ingested = 0
    for did, rec in iter_corpus(args.corpus_jsonl, limit=args.limit_docs):
        batch[did] = rec
        n_seen += 1
        if len(batch) >= max(1, int(args.batch_size)):
            n_ingested += graph.ingest_documents(batch)
            batch.clear()
        if n_seen % max(1, int(args.progress_every)) == 0:
            elapsed = time.perf_counter() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(
                json.dumps(
                    {
                        "progress_docs": n_seen,
                        "graph_docs": len(graph.docs),
                        "graph_sents": len(graph.sentences),
                        "graph_ents": len(graph.entities),
                        "docs_per_sec": round(rate, 1),
                        "elapsed_sec": round(elapsed, 1),
                    }
                ),
                flush=True,
            )
    if batch:
        n_ingested += graph.ingest_documents(batch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_graph_index(graph, args.out)
    graph.source_path = str(args.out)
    meta = validate_loaded_graph(graph, required=True, path=str(args.out))
    report = {
        "ok": True,
        "corpus_jsonl": str(args.corpus_jsonl),
        "graph_index_path": str(args.out),
        "n_jsonl_docs": n_seen,
        "n_ingested": n_ingested,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        **meta,
    }
    report_path = args.out.with_suffix(args.out.suffix + ".METADATA.json")
    if args.out.suffix.lower() == ".pkl":
        report_path = args.out.with_name(args.out.stem + ".METADATA.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if int(meta.get("graph_num_docs") or 0) <= 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
