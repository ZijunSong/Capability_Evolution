#!/usr/bin/env python3
"""Oracle BFS reachability from initial retrieval along a corpus graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.harness1_metrics import set_recall as metric_recall
from trim.eval.harness_g_contract import validate_loaded_graph
from trim.eval.harness_g_graph import load_graph_index


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


def initial_docs(row: dict, searcher, search_k: int, skip_retrieval: bool) -> set[str]:
    preset = {str(x) for x in (row.get("initial_docids") or []) if str(x)}
    if skip_retrieval or searcher is None or getattr(searcher, "name", "none") == "none":
        return preset
    hits = searcher.search(str(row.get("query") or ""), int(search_k)) or []
    return {str(h.docid) for h in hits} | preset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Oracle graph reachability vs initial BM25")
    parser.add_argument("--graph-index-path", required=True)
    parser.add_argument("--benchmark", default="bcplus_test_50")
    parser.add_argument("--queries-jsonl", type=Path, default=None)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-entity-docs", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    graph = load_graph_index(args.graph_index_path)
    validate_loaded_graph(graph, required=True, path=args.graph_index_path)
    rows = load_rows(args.benchmark, args.queries_jsonl)
    searcher = None
    if not args.skip_retrieval and args.queries_jsonl is None:
        from trim.eval.transfer_benchmarks import open_eval_retrieval

        searcher = open_eval_retrieval(args.benchmark, formal=True)

    init_hits = 0
    hop_hits = {h: 0 for h in range(1, args.max_hops + 1)}
    miss_then_reach = 0
    n = 0
    details = []
    for row in rows:
        gold = {str(x) for x in (row.get("gold_docids") or [])}
        if not gold:
            continue
        n += 1
        initial = initial_docs(row, searcher, args.search_k, args.skip_retrieval)
        init_ok = bool(initial & gold)
        init_hits += int(init_ok)
        reached_now = set(initial)
        hop_ok = {}
        for hop in range(1, args.max_hops + 1):
            if reached_now & gold:
                hop_ok[hop] = True
                hop_hits[hop] += 1
                continue
            reached_now = graph.expand_docs(
                reached_now or initial,
                hops=1,
                max_entity_docs=args.max_entity_docs,
                stop_docids=gold,
            ) | reached_now
            hop_ok[hop] = bool(reached_now & gold)
            hop_hits[hop] += int(hop_ok[hop])
        if (not init_ok) and any(hop_ok.values()):
            miss_then_reach += 1
        details.append(
            {
                "query_id": row.get("query_id"),
                "initial_hit": init_ok,
                "hop_hit": hop_ok,
                "initial_recall": metric_recall(initial, gold),
            }
        )
        print(
            json.dumps(
                {
                    "progress": n,
                    "query_id": row.get("query_id"),
                    "initial_hit": init_ok,
                    "hop_hit": hop_ok,
                }
            ),
            flush=True,
        )
    payload = {
        "n_queries": n,
        "initial_recall": init_hits / max(1, n),
        "graph_oracle_recall": {str(h): hop_hits[h] / max(1, n) for h in hop_hits},
        "initial_miss_graph_reachable": miss_then_reach,
        "oracle_beats_initial": (hop_hits[args.max_hops] / max(1, n)) > (init_hits / max(1, n)),
        "max_entity_docs": args.max_entity_docs,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({"summary": payload, "details": details}, indent=2) + "\n", encoding="utf-8")
    if n and not payload["oracle_beats_initial"] and miss_then_reach == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
