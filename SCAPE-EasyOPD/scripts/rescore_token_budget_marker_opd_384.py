#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).with_name("eval_token_budget_marker_opd_384.py")
spec = importlib.util.spec_from_file_location("token_opd_384", RUNNER)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
SETTINGS = ("TEACHER", "STUDENT_BEFORE_OPD", "STUDENT_AFTER_PURE_OPD", "STUDENT_AFTER_RL_PLUS_OPD")


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.root / "384_QUERY_MANIFEST.json").read_text(encoding="utf-8"))
    by_qid = {r["query_id"]: r for r in manifest["queries"]}
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(module.BCP_ROOT / "indexes" / "bm25"))
    summaries = []
    for setting in SETTINGS:
        path = args.root / setting / "PER_QUERY.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if len(rows) != 384 or len({r["query_id"] for r in rows}) != 384:
            raise RuntimeError(f"{setting}: expected 384 unique rows")
        rescored: list[dict[str, Any]] = []
        for idx, old in enumerate(rows, 1):
            source = by_qid[old["query_id"]]
            action = module.parse_action(old["generated_text"])
            query = module.search_query(action, source)
            retrieved = [str(h.docid) for h in searcher.search(query, 1000)] if query and action["legal"] and action["executable"] else []
            gold = {module.norm_doc(x) for x in source["evidence_docids"]}
            recalls = {}
            hits = {}
            for k in (5, 100, 1000):
                hit = len({module.norm_doc(x) for x in retrieved[:k]} & gold)
                hits[str(k)] = hit
                recalls[str(k)] = hit / max(1, len(gold))
            rescored.append({**old, **action, "retrieval_query": query, "retrieved_docids": retrieved, "evidence_hits_at_k": hits, "evidence_recall_at_k": recalls, "evidence_qrel_count": len(gold), "parser_version": "compact_tool_json_v2"})
            if idx % 64 == 0:
                print(json.dumps({"setting": setting, "rescored": idx}), flush=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rescored:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "setting": setting,
            "n_queries": 384,
            "legal_action_rate": mean([float(r["legal"]) for r in rescored]),
            "executable_action_rate": mean([float(r["executable"]) for r in rescored]),
            "retrieval_nonempty_rate": mean([float(bool(r["retrieved_docids"])) for r in rescored]),
            "test_evidence_recall_at_5": mean([r["evidence_recall_at_k"]["5"] for r in rescored]),
            "test_evidence_recall_at_100": mean([r["evidence_recall_at_k"]["100"] for r in rescored]),
            "test_evidence_recall_at_1000": mean([r["evidence_recall_at_k"]["1000"] for r in rescored]),
            "adapter_reload_path": json.loads((args.root / setting / "SUMMARY.json").read_text(encoding="utf-8"))["adapter_reload_path"],
            "parser_version": "compact_tool_json_v2",
        }
        (args.root / setting / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)
    payload = {
        "status": "TOKEN_BUDGET_MARKER_OPD_384_READY",
        "component": "token_budget_marker",
        "query_count": 384,
        "pool_status": manifest["status"],
        "training_overlap_query_ids": manifest["training_overlap_query_ids"],
        "base_model": module.BASE_MODEL,
        "student_inference_privilege": False,
        "qrel": "qrel_evidence.txt",
        "normalization": "split_at_first_underscore_v1",
        "retrieval": "BrowseComp-Plus official BM25 ordered top-1000 docids",
        "settings": summaries,
    }
    (args.root / "SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.root / "SHA256SUMS").write_text("\n".join(f"{module.sha256(p)}  {p.relative_to(args.root)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
