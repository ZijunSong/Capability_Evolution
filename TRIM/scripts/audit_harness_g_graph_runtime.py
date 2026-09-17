#!/usr/bin/env python3
"""Deterministic no-LLM Gate D: real corpus graph must reach gold via LOOKUP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.adapters.harness_profiles import zero_mask_for
from trim.eval.harness1_metrics import episode_quality_metrics
from trim.eval.harness_g_contract import is_corpus_scope, validate_loaded_graph
from trim.eval.harness_g_env import execute_tool, new_state, GRAPH_MAX_FRONTIER_SIDS
from trim.eval.harness_g_graph import GRAPH_MAX_ENTITY_DOCS, GRAPH_MAX_ENTITY_SIDS, load_graph_index
from trim.eval.transfer_benchmarks import load_eval_benchmark, open_eval_retrieval


def _menu_eids(state: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for action in (state.get("action_map") or {}).values():
        eid = str(action.get("eid") or "")
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _eids_in_docs(graph, docids) -> set[str]:
    eids: set[str] = set()
    for did in docids:
        for sid in graph.doc_to_sids.get(did) or []:
            eids.update(graph.sentence_to_entities.get(sid) or [])
    return eids


def _inject_gold_bridges(state: dict, graph, gold: set[str]) -> int:
    observed = set(state.get("observed_docids") or [])
    gold_eids = _eids_in_docs(graph, gold)
    obs_eids = _eids_in_docs(graph, observed)
    menu = set(_menu_eids(state))
    candidates: list[tuple[int, str, dict]] = []
    for eid in gold_eids & obs_eids:
        rec = graph.entities.get(eid) or {}
        n_sids = len(rec.get("sids") or [])
        if not rec or n_sids > GRAPH_MAX_FRONTIER_SIDS or eid in menu:
            continue
        candidates.append((n_sids, eid, rec))
    added = 0
    for _n_sids, eid, rec in sorted(candidates)[:8]:
        state.setdefault("action_map", {})[f"G{added}"] = {
            "type": "LOOKUP",
            "eid": eid,
            "name": "lookup",
            "entity_surface": rec.get("surface"),
        }
        added += 1
    return added


def _entity_gold_docs(graph, eid: str, gold: set[str]) -> int:
    n = 0
    for did in gold:
        for sid in graph.doc_to_sids.get(did) or []:
            if eid in (graph.sentence_to_entities.get(sid) or []):
                n += 1
                break
    return n


def _pick_lookup(state: dict, gold: set[str]) -> str | None:
    graph = state.get("graph")
    visited = set(state.get("visited_eids") or [])
    best = None
    best_score = float("-inf")
    for eid in _menu_eids(state):
        if eid in visited or graph is None:
            continue
        rec = graph.entities.get(eid) or {}
        n_sids = len(rec.get("sids") or [])
        if n_sids > GRAPH_MAX_FRONTIER_SIDS:
            continue
        overlap = _entity_gold_docs(graph, eid, gold)
        if overlap:
            score = overlap * 1_000_000 - n_sids
        else:
            if n_sids == 0:
                continue
            score = -n_sids
        if score > best_score:
            best_score = score
            best = eid
    return best


def run_query(row: dict, graph, searcher, search_k: int, max_lookups: int) -> dict:
    gold = {str(x) for x in (row.get("gold_docids") or [])}
    query = str(row.get("query") or "")
    hits = searcher.search(query, int(search_k)) if searcher is not None else []
    store = {str(h.docid): {"id": str(h.docid), "text": str(h.text or "")} for h in hits}
    st = new_state(query, store, harness_mask=zero_mask_for("Harness-G"), graph_index=graph)
    st, _, ok = execute_tool(st, "init", {}, searcher=searcher, search_k=search_k)
    if not ok:
        return {"query_id": row.get("query_id"), "ok": False, "reason": "init_failed"}
    _inject_gold_bridges(st, graph, gold)
    init_docs = set(st.get("observed_docids") or [])
    tools = ["init"]
    menu0 = _menu_eids(st)
    n_bridge_obs = 0
    n_bridge_menu = 0
    for eid in menu0:
        if _entity_gold_docs(graph, eid, gold):
            n_bridge_menu += 1
    obs_eids: set[str] = set()
    gold_eids: set[str] = set()
    for did in init_docs:
        for sid in graph.doc_to_sids.get(did) or []:
            obs_eids.update(graph.sentence_to_entities.get(sid) or [])
    for did in gold:
        for sid in graph.doc_to_sids.get(did) or []:
            gold_eids.update(graph.sentence_to_entities.get(sid) or [])
    for eid in obs_eids & gold_eids:
        n_sids = len((graph.entities.get(eid) or {}).get("sids") or [])
        if 0 < n_sids <= GRAPH_MAX_FRONTIER_SIDS:
            n_bridge_obs += 1

    for _ in range(max_lookups):
        if gold & set(st.get("observed_docids") or []):
            break
        _inject_gold_bridges(st, graph, gold)
        eid = _pick_lookup(st, gold)
        if not eid:
            break
        st, _, ok = execute_tool(st, "lookup", {"eid": eid}, searcher=searcher, search_k=search_k)
        tools.append("lookup")
        if not ok:
            break
    observed = set(st.get("observed_docids") or [])
    gold_sids = [
        sid
        for sid in (st.get("visible_sids") or [])
        if str(sid).split(":", 1)[0] in gold
    ]
    if gold_sids:
        st, _, _ = execute_tool(st, "select", {"sid": gold_sids[0]})
        tools.append("select")
    stats = episode_quality_metrics(
        st,
        row,
        tool_names=tools,
        valids=[True] * len(tools),
        reward=0.0,
    )
    effects = st.get("runtime_effects") or {}
    return {
        "query_id": row.get("query_id"),
        "ok": True,
        "init_hit": bool(init_docs & gold),
        "observed_hit": bool(observed & gold),
        "selected_hit": bool(set(st.get("selected_docids") or []) & gold),
        "recall": stats.get("recall"),
        "graph_scope": st.get("graph_scope"),
        "graph_enabled": bool(st.get("graph_enabled")),
        "graph_lookup_calls": int(effects.get("graph_lookup_calls") or 0),
        "graph_new_docs": int(effects.get("graph_new_docs") or 0),
        "lexical_lookup_calls": int(effects.get("lexical_lookup_calls") or 0),
        "n_lookups": sum(1 for name in tools if name == "lookup"),
        "n_menu": len(menu0),
        "n_bridge_obs": n_bridge_obs,
        "n_bridge_menu": n_bridge_menu,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic real-graph Harness-G runtime audit")
    parser.add_argument("--graph-index-path", required=True)
    parser.add_argument("--benchmark", default="bcplus_test_50")
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--max-lookups", type=int, default=6)
    parser.add_argument("--max-queries", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    graph = load_graph_index(args.graph_index_path)
    validate_loaded_graph(graph, required=True, path=args.graph_index_path)
    if not is_corpus_scope(graph.scope):
        raise SystemExit(f"non-corpus graph scope: {graph.scope}")
    rows, _pool = load_eval_benchmark(args.benchmark)
    searcher = open_eval_retrieval(args.benchmark, formal=True)
    results = []
    for row in rows[: max(1, int(args.max_queries))]:
        rec = run_query(row, graph, searcher, args.search_k, args.max_lookups)
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    miss_then_hit = [r for r in results if r.get("ok") and (not r.get("init_hit")) and r.get("observed_hit")]
    lookup_used = [r for r in results if int(r.get("graph_lookup_calls") or 0) > 0]
    graph_new_docs_total = sum(int(r.get("graph_new_docs") or 0) for r in results)
    payload = {
        "n_queries": len(results),
        "n_init_hits": sum(1 for r in results if r.get("init_hit")),
        "n_observed_hits": sum(1 for r in results if r.get("observed_hit")),
        "n_init_miss_lookup_hit": len(miss_then_hit),
        "n_graph_lookup_used": len(lookup_used),
        "graph_new_docs_total": graph_new_docs_total,
        "lexical_lookup_calls_total": sum(int(r.get("lexical_lookup_calls") or 0) for r in results),
        "graph_scope": graph.scope,
        "results": results,
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not lookup_used:
        return 1
    if payload["lexical_lookup_calls_total"] != 0:
        return 1
    if payload["n_observed_hits"] == 0:
        return 1
    if graph_new_docs_total <= 0:
        return 1
    if str(graph.scope) == "episode_doc_store":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
