#!/usr/bin/env python3
"""Merge and fully score auto-populate OPD 384-query evaluation shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SETTINGS = ("TEACHER", "STUDENT_BEFORE_OPD", "STUDENT_AFTER_PURE_OPD", "STUDENT_AFTER_RL_PLUS_OPD")
CORPUS = Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text: str) -> set[str]:
    return {x for x in text.lower().replace("_", " ").split() if len(x) > 2}


def load_corpus() -> list[tuple[str, str, set[str]]]:
    docs = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                docid = str(row.get("id") or row.get("docid") or row.get("source"))
                text = str(row.get("text") or row.get("contents") or row.get("content") or "")
                if docid and text:
                    docs.append((docid, text.lower(), tokens(text)))
    return docs


def search(query: str | None, docs: list[tuple[str, str, set[str]]], k: int = 1000) -> list[str]:
    if not query:
        return []
    q = tokens(query)
    lower = query.lower()
    scored = []
    for docid, text, dt in docs:
        score = float(len(q & dt)) + (1.0 if lower in text else 0.0)
        if score > 0:
            scored.append((score, docid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [docid for _, docid in scored[:k]]


def norm(docid: str) -> str:
    return str(docid).split("_", 1)[0]


def recall(ranked: list[str], relevant: set[str], k: int) -> float:
    return len({norm(x) for x in ranked[:k]} & relevant) / max(1, len(relevant))


def ndcg(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    gains = [1.0 if norm(docid) in relevant else 0.0 for docid in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    subset = rows if split == "all_pool" else [r for r in rows if r["official_split"] == split]
    n = len(subset)
    keys = (
        "evidence_recall_at_5", "evidence_recall_at_100", "evidence_recall_at_1000", "evidence_ndcg_at_10",
        "gold_recall_at_5", "gold_recall_at_100", "gold_recall_at_1000", "gold_ndcg_at_10",
    )
    result = {
        "split": split, "n_queries": n,
        "legal_action_rate": sum(bool(r["legal"]) for r in subset) / n,
        "executable_action_rate": sum(bool(r["executable"]) for r in subset) / n,
        "retrieval_nonempty_rate": sum(bool(r["retrieved_docids"]) for r in subset) / n,
    }
    result.update({key: sum(float(r[key]) for r in subset) / n for key in keys})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [args.shards / setting / "384_QUERY_MANIFEST.json" for setting in SETTINGS]
    hashes = [sha256(path) for path in manifests]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"shard manifest mismatch: {hashes}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    qrels = {row["query_id"]: row for row in manifest["queries"]}
    (args.output_dir / "384_QUERY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    docs = load_corpus()
    summaries = []
    ordered_ids = None
    for setting in SETTINGS:
        source = args.shards / setting / setting / "PER_QUERY.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [row["query_id"] for row in rows]
        if len(rows) != 384 or len(set(ids)) != 384:
            raise RuntimeError(f"{setting}: expected 384 unique query rows")
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise RuntimeError(f"{setting}: ordered query IDs differ")
        for i, row in enumerate(rows, 1):
            retrieved = search(row.get("retrieval_query") if row.get("legal") and row.get("executable") else None, docs)
            rel = qrels[row["query_id"]]
            evidence = {norm(x) for x in rel["evidence_docids"]}
            gold = {norm(x) for x in rel["gold_docids"]}
            row["retrieved_docids"] = retrieved
            row["auto_populated_docids"] = retrieved[:10]
            for k in (5, 100, 1000):
                row[f"evidence_recall_at_{k}"] = recall(retrieved, evidence, k)
                row[f"gold_recall_at_{k}"] = recall(retrieved, gold, k)
            row["evidence_ndcg_at_10"] = ndcg(retrieved, evidence)
            row["gold_ndcg_at_10"] = ndcg(retrieved, gold)
            if i % 64 == 0:
                print(json.dumps({"setting": setting, "rescored": i, "n": len(rows)}), flush=True)
        out = args.output_dir / setting
        out.mkdir(parents=True, exist_ok=True)
        per_query = out / "PER_QUERY.jsonl"
        with per_query.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        source_summary = json.loads((args.shards / setting / setting / "SUMMARY.json").read_text(encoding="utf-8"))
        summary = {
            "setting": setting,
            "adapter_reload_path": source_summary["adapter_reload_path"],
            "all_pool": summarize(rows, "all_pool"),
            "official_test": summarize(rows, "test"),
        }
        (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)

    payload = {
        "status": "AUTO_POPULATE_FIRST_SEARCH_OPD_384_COMPLETE",
        "component": "auto_populate_first_search",
        "query_count": 384,
        "test_query_count": 76,
        "manifest_sha256": sha256(args.output_dir / "384_QUERY_MANIFEST.json"),
        "retrieval_backend": "local_corpus_token_overlap",
        "retrieval_backend_note": "System JDK was unavailable; this is the audited SCAPE local-corpus fallback, not official Lucene BM25 parity.",
        "retrieval_depth": 1000,
        "auto_populated_top_k": 10,
        "settings": summaries,
        "student_inference_privilege": False,
    }
    summary_path = args.output_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(args.output_dir)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
