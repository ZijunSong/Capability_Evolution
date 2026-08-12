#!/usr/bin/env python3
"""Build a SCAPE-local BrowseComp+ Chroma backend from local qrel corpus text.

This does not substitute SCOPE BM25 as the retrieval method. It uses the local
BrowseComp-Plus Lucene index only as a document store to recover qrel-aligned
chunk text, then writes a local Chroma PersistentClient collection plus a JSONL
corpus manifest for SCAPE pre-stage experiments.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _hash_embedding(text: str, *, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    tokens = [tok for tok in text.lower().split() if tok]
    if not tokens:
        return vec
    for tok in tokens[:2048]:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--browsecomp-root",
        type=Path,
        default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"),
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_chroma")
    ap.add_argument("--collection", default="scape_browsecompplus_local_test")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    chroma_path = out / "chroma"
    corpus_path = out / "corpus.jsonl"
    report_path = out / "BUILD_REPORT.json"

    manifest = build_run_manifest(
        run_id="browsecomp_local_chroma",
        stage="retrieval_backend_build",
        command=["python", "scripts/build_browsecomp_local_chroma.py"],
        repo_root=REPO,
        output_dir=out,
        extra={
            "browsecomp_root": str(args.browsecomp_root),
            "collection": args.collection,
            "limit_docs": args.limit_docs,
            "embedding": "deterministic_hash_128",
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    qrel_gold = args.browsecomp_root / "topics-qrels" / "qrel_golds.txt"
    qrel_evidence = args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"
    index_path = args.browsecomp_root / "indexes" / "bm25"
    docids = _read_qrel_docids([qrel_gold, qrel_evidence], limit=args.limit_docs)

    # Import after manifest write so dependency failures leave provenance.
    import chromadb
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_path))
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    collection = client.create_collection(args.collection)

    n_found = 0
    n_missing = 0
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, str]] = []
    embs: list[list[float]] = []

    def flush() -> None:
        nonlocal ids, docs, metas, embs
        if not ids:
            return
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        ids, docs, metas, embs = [], [], [], []

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
            record = {"id": docid, "source": docid, "text": text}
            corpus_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ids.append(docid)
            docs.append(text)
            metas.append({"source": docid})
            embs.append(_hash_embedding(text))
            n_found += 1
            if len(ids) >= args.batch_size:
                flush()
            if i % 10 == 0 or i == len(docids):
                write_status_live(
                    out / "STATUS_LIVE.md",
                    stage="retrieval_backend_build",
                    run_id=manifest["run_id"],
                    n_expected=len(docids),
                    n_finished=i,
                    errors=[],
                    extra={"found": n_found, "missing": n_missing},
                )
        flush()

    # Query smoke.
    smoke = collection.query(query_embeddings=[_hash_embedding("evidence topic")], n_results=min(3, max(1, n_found)))
    report = {
        "ok": n_found > 0,
        "browsecomp_root": str(args.browsecomp_root),
        "chroma_path": str(chroma_path),
        "collection": args.collection,
        "n_qrel_docids": len(docids),
        "n_found": n_found,
        "n_missing": n_missing,
        "collection_count": collection.count(),
        "query_smoke_ids": smoke.get("ids", [[]])[0],
        "embedding": "deterministic_hash_128",
        "note": "Local SCAPE backend built from qrel-aligned corpus text; upstream Harness-1 CloudClient still requires cloud Chroma credentials.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_status_live(
        out / "STATUS_LIVE.md",
        stage="retrieval_backend_build",
        run_id=manifest["run_id"],
        n_expected=len(docids),
        n_finished=len(docids),
        errors=[] if report["ok"] else ["no documents indexed"],
        extra={"found": n_found, "missing": n_missing, "collection": args.collection},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0 if report["ok"] else 1, completed_shards=[args.collection] if report["ok"] else []))
    write_sha256sums(out, [out / "RUN_MANIFEST.json", out / "STATUS_LIVE.md", corpus_path, report_path])
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
