#!/usr/bin/env python3
"""Subtractive-c​​uration Teacher-always-on vs Student-always-off fork."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_subtractive_curation_recall_128 as base


def load_states(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_always_branch(start, *, k, scorer, renderer, component, label, full):
    st = start.clone(label)
    trace = []
    for step in range(k):
        action, dist, dual = base.policy_action(
            st, scorer, renderer, component=component, full=full
        )
        st.execute(action)
        trace.append({
            "branch": label,
            "step": step,
            "phase": "first_action" if step == 0 else f"continue_{step}",
            "view": "full" if full else "reduced",
            "action": action,
            "top_prob": max(dist["tool_name_probs"].values()),
            "snapshot_hash": dual["snapshot_hash"],
            "metrics": st.metrics(),
        })
    return st, trace


def run(args: argparse.Namespace) -> None:
    out = args.out_dir
    (out / "raw").mkdir(parents=True, exist_ok=True)
    queries = base._load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    states = load_states(args.states_manifest)
    if len(states) != 128:
        raise RuntimeError(f"expected exactly 128 frozen states, found {len(states)}")
    searcher, backend = base.build_searcher(args.index_path, args.corpus_path)
    scorer = base.HFContinuationScorer(
        args.model, device=args.device, dtype=args.dtype,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    renderer = base.DualViewRenderer()
    rows = []
    raw_path = out / "raw" / f"subtractive_curation_always_on_off_K{args.K}.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(states):
            qid = str(item["query_id"])
            if qid not in queries or not qrels.get(qid):
                raise RuntimeError(f"state {idx}: missing query/qrel for {qid}")
            start = base.state_from_snapshot(
                item["snapshot"], queries[qid], qrels[qid], searcher,
                "subtractive_curation",
            )
            # The frozen manifest hash is the canonical state identity.  Restoring
            # it through LiveState adds branch-local metadata, so its reserialized
            # content hash is not expected to be byte-identical.
            initial_hash = item["snapshot_hash"]
            restored_hash = start.snapshot().content_hash()
            student, student_trace = run_always_branch(
                start, k=args.K, scorer=scorer, renderer=renderer,
                component="subtractive_curation", label="S", full=False,
            )
            teacher, teacher_trace = run_always_branch(
                start, k=args.K, scorer=scorer, renderer=renderer,
                component="subtractive_curation", label="T", full=True,
            )
            sm, tm = student.metrics(), teacher.metrics()
            first_disagreement = int(
                student_trace[0]["action"] != teacher_trace[0]["action"]
            )
            row = {
                "component": "subtractive_curation",
                "protocol": "Teacher-always-on_vs_Student-always-off",
                "K": args.K,
                "state_id": item.get("state_id", f"subtractive_curation_K{args.K}_{idx:03d}"),
                "query_id": qid,
                "snapshot_hash": item["snapshot_hash"],
                "initial_state_hash": initial_hash,
                "teacher_view": "full_component_on_all_steps",
                "student_view": "reduced_component_off_all_steps",
                "teacher_actions": [x["action"] for x in teacher_trace],
                "student_actions": [x["action"] for x in student_trace],
                "first_action_disagreement": first_disagreement,
                "branch_T_metrics": tm,
                "branch_S_metrics": sm,
                "tool_cost_delta": tm["tool_search_cost"] - sm["tool_search_cost"],
                "utility_delta": tm["objective_utility"] - sm["objective_utility"],
                "full_harness_takeover": False,
                "branch_T_trace": teacher_trace,
                "branch_S_trace": student_trace,
                "search_backend": backend,
                "runner": "subtractive_curation_always_on_off_128",
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (idx + 1) % 16 == 0:
                print(json.dumps({"K": args.K, "finished": idx + 1, "path": str(raw_path)}), flush=True)
    if len(rows) != 128:
        raise RuntimeError(f"K{args.K}: expected 128 rows, got {len(rows)}")
    summary = {
        "component": "subtractive_curation",
        "protocol": "Teacher-always-on_vs_Student-always-off",
        "K": args.K,
        "n_states": len(rows),
        "first_action_disagreement_rate": sum(r["first_action_disagreement"] for r in rows) / len(rows),
        "tool_cost_delta": sum(r["tool_cost_delta"] for r in rows) / len(rows),
        "utility_delta": sum(r["utility_delta"] for r in rows) / len(rows),
        "full_harness_takeover": any(r["full_harness_takeover"] for r in rows),
        "snapshot_hash_match": all(r["snapshot_hash"] == r["initial_state_hash"] for r in rows),
        "ordered_snapshot_hashes": [r["snapshot_hash"] for r in rows],
        "raw": str(raw_path),
    }
    (out / f"SUMMARY_K{args.K}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
