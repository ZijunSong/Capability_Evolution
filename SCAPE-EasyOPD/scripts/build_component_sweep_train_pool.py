#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


QWEN3_LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")


def norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_existing(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else payload
    if rows:
        out = []
        for row in rows:
            if isinstance(row, dict):
                qid = str(row.get("query_id", row.get("id", "")))
                query = str(row.get("query") or row.get("question") or row.get("query_text") or qid)
            else:
                qid = str(row)
                query = qid
            out.append({"query_id": qid, "query": query, "source": "existing_train_pool_446"})
        return out
    return [{"query_id": str(q), "query": str(q), "source": "existing_train_pool_446"} for q in payload.get("query_ids", [])]


def load_query_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[str(parts[0])] = parts[1]
    return out


def load_query_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(q) for q in payload.get("query_ids", [])}


def corpus_candidates(path: Path, *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = str(doc.get("id") or doc.get("source") or "")
            text = str(doc.get("text") or "")
            if not doc_id or len(text) < 240:
                continue
            title = ""
            date = ""
            for raw in text.splitlines()[:12]:
                if raw.lower().startswith("title:"):
                    title = raw.split(":", 1)[1].strip()
                if raw.lower().startswith("date:"):
                    date = raw.split(":", 1)[1].strip()
            body = " ".join(x.strip() for x in text.splitlines() if x.strip() and not x.startswith("---"))
            words = body.split()
            if len(words) < 40:
                continue
            span = " ".join(words[20:80])[:800]
            if not title:
                title = " ".join(words[:10])
            variants = [
                ("single_doc_title_date", f"Using the TRAIN corpus document titled '{title}', what concrete event or fact is described around {date or 'the document date'}?"),
                ("single_doc_evidence_span", f"What does the TRAIN corpus document {doc_id} say about {title[:120]}?"),
            ]
            for method, query in variants:
                rows.append({
                    "query_id": f"synth_doc_{doc_id}_{method}",
                    "query": query,
                    "source": "train_corpus_document_synthesis",
                    "construction_method": method,
                    "source_doc_ids": [doc_id],
                    "evidence_spans": [{"doc_id": doc_id, "span": span}],
                    "reference_answer": span[:240],
                    "required_facts": [title] if title else [],
                })
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['query_id']}:{r['query']}".encode()).hexdigest())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing-train", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_TRAIN_POOL.json"))
    ap.add_argument("--dev", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_DEV.json"))
    ap.add_argument("--test", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_TEST.json"))
    ap.add_argument("--queries-tsv", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/topics-qrels/queries.tsv"))
    ap.add_argument("--corpus", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/manifests"))
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    existing = load_existing(args.existing_train)
    dev_ids = load_query_ids(args.dev)
    test_ids = load_query_ids(args.test)
    excluded = dev_ids | test_ids
    source_queries = load_query_tsv(args.queries_tsv)
    seen_ids = {r["query_id"] for r in existing}
    seen_norm = {norm(r["query"]) for r in existing}
    rows: list[dict[str, Any]] = []
    for row in existing:
        if row["query_id"] in excluded:
            continue
        rows.append(row)
    candidates: list[dict[str, Any]] = []
    for qid, query in source_queries.items():
        if qid in excluded or qid in seen_ids or norm(query) in seen_norm:
            continue
        candidates.append({"query_id": qid, "query": query, "source": "train_side_browsecomp_query", "construction_method": "corpus_grounded_existing_query_text", "source_doc_ids": [], "reference_answer": None, "required_facts": []})
    candidates.extend(corpus_candidates(args.corpus, seed=args.seed))
    candidates.sort(key=lambda r: hashlib.sha256(f"{args.seed}:{r['query_id']}:{r['query']}".encode()).hexdigest())
    seen_ids = {r["query_id"] for r in rows}
    seen_norm = {norm(r["query"]) for r in rows}
    for cand in candidates:
        if len(rows) >= args.target:
            break
        if cand["query_id"] in seen_ids or norm(cand["query"]) in seen_norm:
            continue
        if cand["query_id"] in excluded:
            continue
        rows.append(cand)
        seen_ids.add(cand["query_id"])
        seen_norm.add(norm(cand["query"]))
    unique_norm = {norm(r["query"]) for r in rows}
    status = "READY_2000" if len(rows) == args.target and len(unique_norm) == len(rows) else "QUERY_POOL_INSUFFICIENT"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = {
        "schema_version": "component_sweep_train_pool_v2",
        "status": status,
        "logical_model_id": QWEN3_LOGICAL_MODEL_ID,
        "query_count": len(rows),
        "queries": [{**r, "normalized_query_sha256": hashlib.sha256(norm(r["query"]).encode()).hexdigest(), "source_bundle_sha256": sha(r.get("source_doc_ids", []))} for r in rows],
    }
    (args.out_dir / "COMPONENT_SWEEP_TRAIN_POOL.json").write_text(json.dumps(pool, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "COMPONENT_SWEEP_TRAIN_POOL_PROVENANCE.jsonl").open("w", encoding="utf-8") as f:
        for r in pool["queries"]:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    stats = {"status": status, "n_train_pool_unique_queries": len(rows), "n_exact_duplicate_queries": len(rows) - len(unique_norm), "n_dev_test_query_overlap": len({r["query_id"] for r in rows} & excluded), "n_existing_kept": sum(r.get("source") == "existing_train_pool_446" for r in rows), "n_added": sum(r.get("source") != "existing_train_pool_446" for r in rows), "target": args.target}
    (args.out_dir / "COMPONENT_SWEEP_TRAIN_POOL_STATS.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "COMPONENT_SWEEP_QUERY_LEAKAGE_AUDIT.md").write_text("# COMPONENT_SWEEP_QUERY_LEAKAGE_AUDIT\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in stats.items()) + "\n\nNo DEV/TEST query ids were used for generation; DEV/TEST ids were used only as exclusions.\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0 if status == "READY_2000" else 3


if __name__ == "__main__":
    raise SystemExit(main())
