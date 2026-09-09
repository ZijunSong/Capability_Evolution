#!/usr/bin/env python3
"""Export BrowseComp+ document text from the Lucene BM25 index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRIM = Path(__file__).resolve().parents[1]
REPO = TRIM.parent
if str(TRIM) not in sys.path:
    sys.path.insert(0, str(TRIM))

DEFAULT_BCP = REPO / "SCOPE" / "external" / "BrowseComp-Plus"


def _read_qrel_docids(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                docid = parts[2]
                if docid in seen:
                    continue
                seen.add(docid)
                out.append(docid)
    return out


def _iter_index_docids(index_path: Path):
    from trim.eval.browsecomp_retrieval import _configure_java_runtime, lucene_stored_text

    _configure_java_runtime()
    from pyserini.index.lucene import LuceneIndexReader

    reader = LuceneIndexReader(str(index_path))
    stats = reader.stats() or {}
    n_docs = int(stats.get("documents") or 0)
    for internal_id in range(n_docs):
        collection_id = reader.convert_internal_docid_to_collection_docid(internal_id)
        raw = reader.doc(collection_id).raw()
        text = lucene_stored_text(raw)
        yield collection_id, text, internal_id, n_docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BrowseComp+ corpus JSONL from Lucene index")
    parser.add_argument("--browsecomp-root", type=Path, default=DEFAULT_BCP)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_BCP / "data" / "browsecomp_plus_corpus_full.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "qrel"),
        default="full",
        help="full enumerates the entire Lucene index; qrel exports qrel docids only (debug)",
    )
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args()

    from trim.eval.browsecomp_retrieval import PyseriniBackend, lucene_stored_text

    index_path = args.browsecomp_root / "indexes" / "bm25"
    if not index_path.is_dir():
        raise SystemExit(f"missing index: {index_path}")

    backend = PyseriniBackend(index_path)
    index_num_docs = int(backend.num_docs())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "qrel":
        qrel_gold = args.browsecomp_root / "topics-qrels" / "qrel_golds.txt"
        qrel_evidence = args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"
        docids = _read_qrel_docids([qrel_gold, qrel_evidence])
        if args.limit_docs is not None:
            docids = docids[: int(args.limit_docs)]
        n_found = 0
        n_missing = 0
        exported_ids: set[str] = set()
        with args.out.open("w", encoding="utf-8") as handle:
            for docid in docids:
                raw = backend.get_doc(docid) or ""
                text = lucene_stored_text(raw)
                if not text.strip():
                    n_missing += 1
                    continue
                handle.write(json.dumps({"id": docid, "source": docid, "text": text}, ensure_ascii=False) + "\n")
                exported_ids.add(docid)
                n_found += 1
        report = {
            "ok": n_found > 0,
            "mode": args.mode,
            "corpus_path": str(args.out),
            "index_path": str(index_path),
            "index_num_docs": index_num_docs,
            "n_requested_docids": len(docids),
            "n_exported": n_found,
            "n_missing_text": n_missing,
            "n_exported_unique": len(exported_ids),
            "index_corpus_delta": len(exported_ids) - index_num_docs,
            "probe_docids_present": {
                did: did in exported_ids for did in ("59931", "69324", "44797")
            },
            "notes": ["qrel mode is for debugging only; formal eval must use --mode full"],
        }
    else:
        n_found = 0
        n_missing = 0
        exported_ids: set[str] = set()
        limit = args.limit_docs
        with args.out.open("w", encoding="utf-8") as handle:
            for docid, text, internal_id, n_docs in _iter_index_docids(index_path):
                if limit is not None and n_found >= int(limit):
                    break
                if not text.strip():
                    n_missing += 1
                    continue
                handle.write(json.dumps({"id": docid, "source": docid, "text": text}, ensure_ascii=False) + "\n")
                exported_ids.add(docid)
                n_found += 1
                if n_found % 5000 == 0:
                    print(
                        json.dumps(
                            {
                                "progress": n_found,
                                "index_num_docs": n_docs,
                                "latest_docid": docid,
                                "internal_id": internal_id,
                            }
                        ),
                        flush=True,
                    )
        report = {
            "ok": n_found == index_num_docs - n_missing and len(exported_ids) == n_found,
            "mode": args.mode,
            "corpus_path": str(args.out),
            "index_path": str(index_path),
            "index_num_docs": index_num_docs,
            "n_exported": n_found,
            "n_missing_text": n_missing,
            "n_exported_unique": len(exported_ids),
            "index_corpus_delta": len(exported_ids) - index_num_docs,
            "probe_docids_present": {
                did: did in exported_ids for did in ("59931", "69324", "44797")
            },
            "notes": ["full mode exports every indexed document; qrels are not used"],
        }

    report_path = args.out.with_suffix(".BUILD_REPORT.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.mode == "full" and report["index_corpus_delta"] != 0:
        return 1
    return 0 if n_found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
