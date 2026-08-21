#!/usr/bin/env python3
"""Collect real on-policy first-search -> AUTO projected curate states."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scape.rendering.dual_view import DualViewRenderer
from scripts.run_h100_2_live_fork_replay import HFContinuationScorer, LiveState, _load_qrels, _load_queries, build_searcher, policy_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browsecomp-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"))
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--index-path", type=Path, required=True)
    ap.add_argument("--corpus-path", type=Path, default=REPO / "outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-queries", type=int, default=128)
    ap.add_argument("--seed", type=int, default=8181)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    root = args.browsecomp_root
    queries = _load_queries(root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(root / "topics-qrels" / "qrel_evidence.txt")
    eligible = sorted(set(queries) & set(qrels), key=lambda q: hashlib.sha256(f"projected:{args.seed}:{q}".encode()).hexdigest())[: args.n_queries]
    eligible = eligible[args.shard_index::args.num_shards]
    searcher, backend = build_searcher(args.index_path, args.corpus_path)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype="bfloat16", max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    rows: list[dict[str, Any]] = []
    for idx, qid in enumerate(eligible):
        st = LiveState(qid=qid, query=queries[qid], gold=qrels[qid], searcher=searcher, component="auto_populate_first_search", branch_seed=f"projected:{args.seed}:{qid}")
        # LiveState performs the real first BM25 search in __init__. The pre-hook
        # state is that returned observation with the automatic curated mutation
        # withheld; no model action or hidden document is invented here.
        result_ids = [str(d["id"]) for d in st.documents]
        first_search_action = {"name": "search_corpus", "arguments": {"query": queries[qid]}}
        st.history = [{"step": 0, "action": first_search_action}]
        st.step = 1
        st.curated_ids = []
        st.cost = 1
        pre = st.snapshot()
        pre_curated = list(st.curated_ids)
        auto_ids = []
        for did in result_ids:
            if did not in st.curated_ids and did not in auto_ids:
                auto_ids.append(did)
            if len(auto_ids) >= args.top_k:
                break
        if not auto_ids:
            continue
        st.curated_ids.extend(auto_ids)
        full = st.snapshot()
        dual_full = renderer.render_pair(pre, component_id="auto_populate_first_search", include_null_controls=False)
        next_s, next_s_dist, next_dual = policy_action(st, scorer, renderer, component="auto_populate_first_search", full=False)
        next_t, next_t_dist, _ = policy_action(st, scorer, renderer, component="auto_populate_first_search", full=True)
        add_ids = [x for x in st.curated_ids if x not in pre_curated]
        if not add_ids or not set(add_ids).issubset(set(result_ids)):
            continue
        rows.append({
            "row_id": f"projected_{qid}_{pre.step}_{pre.content_hash()}",
            "query_id": str(qid),
            "step": int(pre.step),
            "snapshot_hash": pre.content_hash(),
            "search_action": first_search_action,
            "search_distribution": {},
            "s_pre": pre.to_dict(),
            "s_full": full.to_dict(),
            "curated_ids_pre": pre_curated,
            "curated_ids_post": list(st.curated_ids),
            "search_result_ids": result_ids,
            "reduced_view": dual_full.student_view,
            "full_view": dual_full.full_view,
            "prompt_reduced": json.dumps(dual_full.student_view, ensure_ascii=False, sort_keys=True),
            "prompt_full": json.dumps(dual_full.full_view, ensure_ascii=False, sort_keys=True),
            "projected_action": {"tool_name": "curate", "parameters": {"add_ids": add_ids, "remove_ids": []}},
            "next_state": full.to_dict(),
            "next_prompt_reduced": json.dumps(next_dual["student_view"], ensure_ascii=False, sort_keys=True),
            "next_prompt_full": json.dumps(next_dual["full_view"], ensure_ascii=False, sort_keys=True),
            "next_student_action": next_s,
            "next_teacher_action": next_t,
            "next_student_tool_distribution": next_s_dist.get("tool_name_probs", {}),
            "next_teacher_tool_distribution": next_t_dist.get("tool_name_probs", {}),
            "provenance": {"query_id": str(qid), "state_hash": pre.content_hash(), "search_result_ids": result_ids, "pre_curated_ids": pre_curated, "post_curated_ids": list(st.curated_ids), "projected_add_ids": add_ids, "projection_source": "deterministic_runtime_state_delta", "student_visible_ids_only": True, "full_harness_takeover": False},
            "student_inference_privilege": False,
            "runner": "real_on_policy_student_first_search_projection",
            "search_backend": backend,
        })
        if len(rows) % 8 == 0:
            (args.out.parent / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- status: collecting\n- finished_positive: {len(rows)}\n- queries_seen: {idx + 1}\n- gpu: {args.device}\n", encoding="utf-8")
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "completed", "n_queries": len(eligible), "n_positive": len(rows), "path": str(args.out), "backend": backend}, ensure_ascii=False))


if __name__ == "__main__":
    main()
