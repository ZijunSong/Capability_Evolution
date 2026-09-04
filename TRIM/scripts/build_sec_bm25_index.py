#!/usr/bin/env python3
"""Build a local Pyserini Lucene BM25 index for the SEC training corpus.

Reads ``{sec_corpus_root}/corpora/sec/train/*.parquet`` and writes
``{sec_corpus_root}/indexes/bm25``. TRIM training uses this path when present.

Example:
  PYTHONPATH=TRIM python TRIM/scripts/build_sec_bm25_index.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.browsecomp_retrieval import _configure_java_runtime
from trim.eval.sec_corpus import corpus_bm25_index, corpus_parquet_dir, default_sec_corpus_root


def _is_lucene_index(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.name.startswith("segments_") and p.is_file() for p in path.iterdir())


def _write_jsonl(parquet_dir: Path, jsonl_dir: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    jsonl_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {parquet_dir}")
    n_docs = 0
    n_empty = 0
    seen: set[str] = set()
    n_dup = 0
    for shard in shards:
        out = jsonl_dir / f"{shard.stem}.json"
        table = pq.read_table(shard, columns=["chunk_id", "document_text"])
        with out.open("w", encoding="utf-8") as handle:
            for cid, text in zip(table["chunk_id"].to_pylist(), table["document_text"].to_pylist()):
                docid = str(cid or "").strip()
                body = str(text or "")
                if not docid:
                    n_empty += 1
                    continue
                if docid in seen:
                    n_dup += 1
                    continue
                seen.add(docid)
                if not body.strip():
                    n_empty += 1
                handle.write(json.dumps({"id": docid, "contents": body}, ensure_ascii=False) + "\n")
                n_docs += 1
        print(f"wrote {out.name}: running_docs={n_docs}", flush=True)
    return {"n_docs": n_docs, "n_empty": n_empty, "n_dup": n_dup, "n_shards": len(shards)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sec-corpus-root", type=Path, default=default_sec_corpus_root())
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 8)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-jsonl", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    root = Path(args.sec_corpus_root)
    parquet_dir = corpus_parquet_dir(root)
    index_dir = corpus_bm25_index(root)
    jsonl_dir = root / "collections" / "jsonl"
    if _is_lucene_index(index_dir) and not args.force:
        print(f"index already present: {index_dir}")
        return 0
    if not parquet_dir.is_dir():
        raise SystemExit(f"SEC parquet corpus missing: {parquet_dir}")

    _configure_java_runtime()
    os.environ.setdefault("OPENAI_API_KEY", "sk-pyserini-local")
    if not os.environ.get("JAVA_HOME"):
        raise SystemExit("JAVA_HOME not set and no JDK was discovered for Pyserini")

    print(f"parquet:   {parquet_dir}")
    print(f"jsonl:     {jsonl_dir}")
    print(f"index:     {index_dir}")
    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
    print(f"threads:   {args.threads}", flush=True)

    stats = _write_jsonl(parquet_dir, jsonl_dir)
    print(f"jsonl stats: {stats}", flush=True)
    if stats["n_docs"] <= 0:
        raise SystemExit("jsonl conversion produced zero documents")

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        "-m",
        "pyserini.index.lucene",
        "-collection",
        "JsonCollection",
        "-input",
        str(jsonl_dir),
        "-index",
        str(index_dir),
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        str(int(args.threads)),
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw",
        "-uniqueDocid",
    ]
    print("indexing:", " ".join(cmd), flush=True)
    env = dict(os.environ)
    subprocess.run(cmd, check=True, env=env)
    if not _is_lucene_index(index_dir):
        raise SystemExit(f"Lucene index was not created at {index_dir}")

    from trim.eval.sec_corpus import open_sec_retrieval

    backend = open_sec_retrieval(root)
    hits = backend.search("Cenntro 8-K director", k=3)
    (index_dir / "BUILD.json").write_text(
        json.dumps(
            {
                "corpus_root": str(root),
                "parquet_dir": str(parquet_dir),
                "n_docs": stats["n_docs"],
                "n_empty": stats["n_empty"],
                "n_dup": stats["n_dup"],
                "backend": backend.name,
                "smoke_hits": [h.docid for h in hits],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.keep_jsonl:
        shutil.rmtree(jsonl_dir, ignore_errors=True)
    print(f"backend={backend.name} smoke_hits={[h.docid for h in hits]}")
    print(f"BM25 index ready: {index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
