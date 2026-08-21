#!/usr/bin/env python3
"""Always-on paired fork for importance_tagging + subtractive_curation.

Teacher uses the Full view for every action in the K-step horizon; Student
uses the Reduced view for every action. The first forced action counts toward K.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_joint_importance_subtractive_preopd_fork import (  # noqa: E402
    HFContinuationScorer,
    JointLiveState,
    _load_qrels,
    _load_queries,
    build_searcher,
    policy_action,
    state_from_snapshot,
)

COMPONENT = "importance_tagging+subtractive_curation"
DEFAULT_STATES = ROOT / "outputs/0820_subtractive_curation_recall_128_final/manifests/SUBTRACTIVE_STATES_128.jsonl"
DEFAULT_OUT = ROOT / "outputs/0821_joint_importance_subtractive_always_on_128"


def action_key(action):
    return json.dumps(action or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run_branch(start, first, *, k, scorer, renderer, full, label):
    st = start.clone(label)
    trace = []
    st.execute(first)
    trace.append({"phase": "step_1", "action": dict(first), "metrics": st.metrics()})
    for step in range(1, k):
        action, dist, dual = policy_action(st, scorer, renderer, full=full)
        st.execute(action)
        trace.append({
            "phase": f"step_{step + 1}",
            "action": action,
            "top_prob": max(dist["tool_name_probs"].values()),
            "snapshot_hash": dual["snapshot_hash"],
            "metrics": st.metrics(),
        })
    return st, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=DEFAULT_STATES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--browsecomp-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"))
    ap.add_argument("--index-path", type=Path, default=None)
    ap.add_argument("--corpus-path", type=Path, default=ROOT / "outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--Ks", type=int, nargs="+", default=[4, 8])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    states = [json.loads(line) for line in args.states.read_text().splitlines() if line.strip()]
    if len(states) != 128 or len({r["snapshot_hash"] for r in states}) != 128:
        raise RuntimeError(f"expected 128 unique frozen snapshots, got {len(states)}")
    index_path = args.index_path or (args.browsecomp_root / "indexes" / "bm25")
    searcher, backend = build_searcher(index_path, args.corpus_path)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    # Imported lazily to keep this script's dependency surface identical to the validated runner.
    from scape.rendering.dual_view import DualViewRenderer
    renderer = DualViewRenderer()

    all_rows = []
    for k in args.Ks:
        for idx, item in enumerate(states):
            qid = str(item["query_id"])
            if qid not in queries or qid not in qrels:
                raise RuntimeError(f"missing query/qrel for {qid}")
            start = state_from_snapshot(item["snapshot"], queries[qid], qrels[qid], searcher)
            # Generate both first actions from the identical initial state.
            a_s, _, _ = policy_action(start, scorer, renderer, full=False)
            a_t, _, _ = policy_action(start, scorer, renderer, full=True)
            s_final, s_trace = run_branch(start, a_s, k=k, scorer=scorer, renderer=renderer, full=False, label=f"S_K{k}_{idx}")
            t_final, t_trace = run_branch(start, a_t, k=k, scorer=scorer, renderer=renderer, full=True, label=f"T_K{k}_{idx}")
            sm, tm = s_final.metrics(), t_final.metrics()
            row = {
                "component": COMPONENT, "protocol": "teacher_always_on_student_always_off",
                "K": k, "state_id": f"joint_always_on_K{k}_{idx:03d}", "query_id": qid,
                "snapshot_hash": item["snapshot_hash"], "a_S": a_s, "a_T": a_t,
                "first_action_disagreement": int(action_key(a_s) != action_key(a_t)),
                "branch_S_metrics": sm, "branch_T_metrics": tm,
                "tool_cost_delta": tm["tool_search_cost"] - sm["tool_search_cost"],
                "utility_delta": tm["objective_utility"] - sm["objective_utility"],
                "full_harness_takeover": False, "search_backend": backend,
                "branch_S_trace": s_trace, "branch_T_trace": t_trace,
            }
            all_rows.append(row)

    summary = {"component": COMPONENT, "protocol": "Teacher Full every step; Student Reduced every step; first action counts toward K", "n_states": 128, "horizons": {}}
    for k in args.Ks:
        rows = [r for r in all_rows if r["K"] == k]
        summary["horizons"][f"K{k}"] = {
            "n": len(rows),
            "first_action_disagreement_rate": statistics.mean(r["first_action_disagreement"] for r in rows),
            "tool_cost_delta": statistics.mean(r["tool_cost_delta"] for r in rows),
            "utility_delta": statistics.mean(r["utility_delta"] for r in rows),
            "positive_negative_zero_utility": [sum(r["utility_delta"] > 0 for r in rows), sum(r["utility_delta"] < 0 for r in rows), sum(r["utility_delta"] == 0 for r in rows)],
        }
    (args.out_dir / "JOINT_ALWAYS_ON_PER_STATE.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    (args.out_dir / "JOINT_ALWAYS_ON_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
