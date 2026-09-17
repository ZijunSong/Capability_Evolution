#!/usr/bin/env python3
"""Static audit of a Harness-G corpus graph against gold docs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.harness_g_contract import is_corpus_scope, validate_loaded_graph
from trim.eval.harness_g_graph import load_graph_index


def normalize(docid: str) -> str:
    return str(docid).strip()


def load_rows(benchmark: str, queries_jsonl: Path | None) -> list[dict]:
    if queries_jsonl is not None:
        rows: list[dict] = []
        with queries_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    from trim.eval.transfer_benchmarks import load_eval_benchmark

    rows, _pool = load_eval_benchmark(benchmark)
    return list(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Harness-G corpus graph coverage")
    parser.add_argument("--graph-index-path", required=True)
    parser.add_argument("--benchmark", default="bcplus_test_50")
    parser.add_argument("--queries-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    graph = load_graph_index(args.graph_index_path)
    meta = validate_loaded_graph(graph, required=True, path=args.graph_index_path)
    rows = load_rows(args.benchmark, args.queries_jsonl)
    gold: set[str] = set()
    evidence: set[str] = set()
    for row in rows:
        gold.update(normalize(x) for x in (row.get("gold_docids") or []))
        evidence.update(normalize(x) for x in (row.get("evidence_docids") or []))
    graph_ids = {normalize(x) for x in graph.docs} | {normalize(x) for x in graph.parent_docids.values()}
    missing_gold = sorted(gold - graph_ids)
    missing_evidence = sorted(evidence - graph_ids)
    payload = {
        **meta,
        "benchmark": args.benchmark,
        "n_queries": len(rows),
        "gold_docs_total": len(gold),
        "gold_docs_in_graph": len(gold) - len(missing_gold),
        "gold_doc_coverage": (len(gold) - len(missing_gold)) / max(1, len(gold)),
        "missing_gold_docids": missing_gold,
        "evidence_docs_total": len(evidence),
        "evidence_docs_in_graph": len(evidence) - len(missing_evidence),
        "missing_evidence_docids": missing_evidence[:50],
        "docid_namespace_ok": not any(
            ":" in did and did.split(":", 1)[0] not in {"http", "https"} and did not in graph_ids
            for did in list(gold)[:20]
        ),
        "corpus_scope_ok": is_corpus_scope(meta.get("graph_scope")),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if not payload["corpus_scope_ok"] or payload["gold_doc_coverage"] < 0.99:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
