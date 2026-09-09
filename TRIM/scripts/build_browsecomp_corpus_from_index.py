#!/usr/bin/env python3
"""Export qrel-aligned BrowseComp+ document text from the Lucene BM25 index."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BrowseComp+ corpus JSONL from Lucene index")
    parser.add_argument("--browsecomp-root", type=Path, default=DEFAULT_BCP)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_BCP / "data" / "browsecomp_plus_corpus.jsonl",
    )
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args()

    from trim.eval.browsecomp_retrieval import PyseriniBackend, lucene_stored_text

    index_path = args.browsecomp_root / "indexes" / "bm25"
    qrel_gold = args.browsecomp_root / "topics-qrels" / "qrel_golds.txt"
    qrel_evidence = args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"
    if not index_path.is_dir():
        raise SystemExit(f"missing index: {index_path}")

    docids = _read_qrel_docids([qrel_gold, qrel_evidence])
    if args.limit_docs is not None:
        docids = docids[: int(args.limit_docs)]

    backend = PyseriniBackend(index_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_found = 0
    n_missing = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for docid in docids:
            raw = backend.get_doc(docid) or ""
            text = lucene_stored_text(raw)
            if not text.strip():
                n_missing += 1
                continue
            handle.write(json.dumps({"id": docid, "source": docid, "text": text}, ensure_ascii=False) + "\n")
            n_found += 1

    report = {
        "ok": n_found > 0,
        "corpus_path": str(args.out),
        "index_path": str(index_path),
        "n_qrel_docids": len(docids),
        "n_found": n_found,
        "n_missing": n_missing,
    }
    report_path = args.out.with_suffix(".BUILD_REPORT.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if n_found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
