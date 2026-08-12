#!/usr/bin/env python3
"""Export qrel-aligned BrowseComp+ corpus text for SCAPE local experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live


def _read_qrel_docids(paths: Iterable[Path], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                docid = parts[2]
                if docid in seen:
                    continue
                seen.add(docid)
                out.append(docid)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--browsecomp-root",
        type=Path,
        default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"),
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_corpus")
    ap.add_argument("--limit-docs", type=int, default=None)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / "corpus.jsonl"
    report_path = out / "BUILD_REPORT.json"

    manifest = build_run_manifest(
        run_id="browsecomp_local_corpus",
        stage="retrieval_corpus_export",
        command=["python", "scripts/build_browsecomp_local_corpus.py"],
        repo_root=REPO,
        output_dir=out,
        extra={"browsecomp_root": str(args.browsecomp_root), "limit_docs": args.limit_docs},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    qrel_gold = args.browsecomp_root / "topics-qrels" / "qrel_golds.txt"
    qrel_evidence = args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"
    index_path = args.browsecomp_root / "indexes" / "bm25"
    docids = _read_qrel_docids([qrel_gold, qrel_evidence], limit=args.limit_docs)

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_path))
    n_found = 0
    n_missing = 0
    with corpus_path.open("w", encoding="utf-8") as corpus_f:
        for i, docid in enumerate(docids, start=1):
            doc = searcher.doc(docid)
            if doc is None:
                n_missing += 1
                continue
            raw = json.loads(doc.raw())
            text = raw.get("contents") or raw.get("text") or ""
            if not text:
                n_missing += 1
                continue
            corpus_f.write(json.dumps({"id": docid, "source": docid, "text": text}, ensure_ascii=False) + "\n")
            n_found += 1
            if i % 1000 == 0 or i == len(docids):
                write_status_live(
                    out / "STATUS_LIVE.md",
                    stage="retrieval_corpus_export",
                    run_id=manifest["run_id"],
                    n_expected=len(docids),
                    n_finished=i,
                    errors=[],
                    extra={"found": n_found, "missing": n_missing},
                )
    report = {
        "ok": n_found > 0,
        "browsecomp_root": str(args.browsecomp_root),
        "corpus_path": str(corpus_path),
        "n_qrel_docids": len(docids),
        "n_found": n_found,
        "n_missing": n_missing,
        "source": "BrowseComp-Plus Lucene stored raw.contents",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0 if report["ok"] else 1, completed_shards=["corpus"] if report["ok"] else []))
    write_sha256sums(out, [out / "RUN_MANIFEST.json", out / "STATUS_LIVE.md", corpus_path, report_path])
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
