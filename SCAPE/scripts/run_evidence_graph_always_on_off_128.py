#!/usr/bin/env python3
"""Evidence-graph always-on/off paired utility fork over frozen 128 states."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence_graph_recall_formal_fork as base


def load_states(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(args):
    out = args.out_dir
    (out / "raw").mkdir(parents=True, exist_ok=True)
    queries = base._load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    states = load_states(args.states_manifest)
    if len(states) != 128:
        raise RuntimeError(f"expected 128 frozen states, found {len(states)}")
    searcher, backend = base.build_searcher(args.index_path, args.corpus_path)
    scorer = base.HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = base.DualViewRenderer()
    rows = []
    path = out / "raw" / f"evidence_graph_always_on_off_K{args.K}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(states):
            qid = str(item["query_id"])
            start = base.state_from_snapshot(item["snapshot"], queries[qid], qrels[qid], searcher, "evidence_graph")
            initial_hash = start.snapshot().content_hash()
            s_final, s_trace = base.run_branch(start, item["a_S"], k=args.K, scorer=scorer, renderer=renderer, component="evidence_graph", label="S", continuation_full=False)
            t_final, t_trace = base.run_branch(start, item["a_T"], k=args.K, scorer=scorer, renderer=renderer, component="evidence_graph", label="T", continuation_full=True)
            sm, tm = s_final.metrics(), t_final.metrics()
            actions_s = [x["action"] for x in s_trace]
            actions_t = [x["action"] for x in t_trace]
            row = {
                "component": "evidence_graph", "protocol": "Teacher-always-on_vs_Student-always-off",
                "K": args.K, "state_id": item.get("state_id", f"evidence_graph_K{args.K}_{idx:03d}"), "query_id": qid,
                "snapshot_hash": item["snapshot_hash"], "initial_state_hash": initial_hash,
                "teacher_view": "full_component_on_all_steps", "student_view": "reduced_component_off_all_steps",
                "teacher_actions": actions_t, "student_actions": actions_s,
                "first_action_disagreement": int(actions_t[0] != actions_s[0]),
                "branch_T_metrics": tm, "branch_S_metrics": sm,
                "tool_cost_delta": tm["tool_search_cost"] - sm["tool_search_cost"],
                "utility_delta": tm["objective_utility"] - sm["objective_utility"],
                "full_harness_takeover": False, "search_backend": backend,
                "branch_T_trace": t_trace, "branch_S_trace": s_trace,
            }
            rows.append(row); f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (idx + 1) % 16 == 0:
                print(json.dumps({"K": args.K, "finished": idx + 1, "path": str(path)}), flush=True)
    n = len(rows)
    summary = {
        "component": "evidence_graph", "protocol": "Teacher-always-on_vs_Student-always-off",
        "K": args.K, "n_states": n,
        "first_action_disagreement_rate": sum(r["first_action_disagreement"] for r in rows) / n,
        "tool_cost_delta": sum(r["tool_cost_delta"] for r in rows) / n,
        "utility_delta": sum(r["utility_delta"] for r in rows) / n,
        "full_harness_takeover": any(r["full_harness_takeover"] for r in rows),
        "snapshot_hash_match": all(r["snapshot_hash"] == r["initial_state_hash"] for r in rows),
        "raw": str(path),
    }
    (out / f"SUMMARY_K{args.K}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, choices=[4, 8], required=True)
    ap.add_argument("--states-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--corpus-path", type=Path, default=HERE.parent / "outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    run(ap.parse_args())

if __name__ == "__main__":
    main()
