#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from scape.common.hashing import stable_split
from scape.rendering.dual_view import DualViewRenderer
from run_joint_importance_subtractive_preopd_fork import (
    HFContinuationScorer,
    JointLiveState,
    _load_qrels,
    _load_queries,
    build_searcher,
    joint_student_mask,
    policy_action,
)

COMPONENT = "importance_tagging_plus_subtractive_curation"


class ProvenanceState(JointLiveState):
    def clone(self, suffix: str) -> "ProvenanceState":
        new = object.__new__(ProvenanceState)
        new.qid = self.qid
        new.query = self.query
        new.gold = set(self.gold)
        new.searcher = self.searcher
        new.component = self.component
        new.branch_seed = f"{self.branch_seed}:{suffix}"
        new.step = self.step
        new.documents = deepcopy(self.documents)
        new.curated_ids = list(self.curated_ids)
        new.read_ids = list(self.read_ids)
        new.successful_read_ids = list(self.successful_read_ids)
        new.read_ids_entered_context = list(self.read_ids_entered_context)
        new.read_ids_retained_at_endpoint = list(self.read_ids_retained_at_endpoint)
        new.verified_supported = list(self.verified_supported)
        new.verified_unsupported = list(self.verified_unsupported)
        new.history = deepcopy(self.history)
        new.observations = deepcopy(self.observations)
        new.cost = self.cost
        return new

    def execute(self, action: dict[str, Any]) -> None:
        before = len(self.observations)
        super().execute(action)
        name = str(action.get("name") or "end_search")
        args = dict(action.get("arguments") or {})
        if name in {"read_document", "review_docs"} and len(self.observations) > before:
            ids = [str(args.get("doc_id") or "")] if name == "read_document" else [str(x) for x in args.get("doc_ids") or []]
            docs = {str(d.get("id")): str(d.get("text") or "") for d in self.documents}
            successful = [did for did in ids if did in docs]
            self.observations[-1].update({
                "tool": name,
                "successful_read_ids": successful,
                "document_observations": [{"id": did, "text": docs[did]} for did in successful],
                "entered_context": bool(successful),
                "retained_at_endpoint": bool(successful),
            })

    def snapshot(self):
        snap = super().snapshot()
        wm = dict(snap.working_memory)
        docs = {str(d.get("id")): d for d in self.documents}
        wm["retained_read_docs"] = [docs[x] for x in self.read_ids_retained_at_endpoint if x in docs]
        wm["retained_read_ids"] = list(self.read_ids_retained_at_endpoint)
        return type(snap)(
            query_id=snap.query_id,
            step=snap.step,
            harness_mask=snap.harness_mask,
            working_memory=wm,
            tool_history=snap.tool_history,
            observations=snap.observations,
            metadata=snap.metadata,
        )


def endpoint(st: ProvenanceState, initial: ProvenanceState) -> dict[str, Any]:
    return {
        "initial_candidate_evidence_ids": [str(d["id"]) for d in initial.documents],
        "final_candidate_evidence_ids": [str(d["id"]) for d in st.documents],
        "initial_curated_ids": list(initial.curated_ids),
        "final_curated_ids": list(st.curated_ids),
        "read_attempt_ids_within_k": list(st.read_ids),
        "successful_read_ids_within_k": list(st.successful_read_ids),
        "read_ids_entered_context": list(st.read_ids_entered_context),
        "read_ids_retained_at_endpoint": list(st.read_ids_retained_at_endpoint),
        "final_activated_evidence_ids": sorted(set(st.curated_ids) | set(st.read_ids_retained_at_endpoint)),
        "context_evidence_ids_by_step": [list(st.read_ids_retained_at_endpoint)],
        "final_state_hash": st.snapshot().content_hash(),
        "tool_cost": st.cost - initial.cost,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--n-states", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    queries = _load_queries(bcp / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(bcp / "topics-qrels" / "qrel_evidence.txt")
    eligible = sorted(qid for qid in set(queries) & set(qrels) if qrels[qid])
    selected, _ = stable_split(eligible, seed=args.seed, n_take=args.n_states)
    selected = list(selected)
    qids = selected[args.shard_index :: args.n_shards]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.out_dir / "manifests"
    shard_dir = args.out_dir / "shards"
    manifest_dir.mkdir(exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    master = {
        "schema_version": "joint_recall_fresh_cohort_v1",
        "seed": args.seed,
        "n_states": len(selected),
        "query_ids": selected,
        "eligibility": "nonempty qrel, initial candidate pool and curated set; both components have opportunity to act",
    }
    (manifest_dir / f"FRESH_COHORT_seed{args.seed}.json").write_text(json.dumps(master, indent=2) + "\n")

    corpus = REPO / "outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl"
    searcher, backend = build_searcher(bcp / "indexes/bm25", corpus)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=3072)
    renderer = DualViewRenderer()
    output = shard_dir / f"fresh_seed{args.seed}_shard{args.shard_index:02d}.jsonl"

    with output.open("w", encoding="utf-8") as f:
        for local_index, qid in enumerate(qids):
            base = ProvenanceState(qid=qid, query=queries[qid], gold=qrels[qid], searcher=searcher, branch_seed=f"fresh:{args.seed}:{qid}")
            initial = base.clone("initial")
            initial_snapshot = base.snapshot()
            a_s, d_s, dual_s = policy_action(base, scorer, renderer, full=False)
            a_t, d_t, dual_t = policy_action(base, scorer, renderer, full=True)
            if dual_s["snapshot_hash"] != dual_t["snapshot_hash"] or dual_s["snapshot_hash"] != initial_snapshot.content_hash():
                raise RuntimeError(f"initial snapshot mismatch for {qid}")
            branches = {}
            for label, first in (("S", a_s), ("T", a_t)):
                st = base.clone(label)
                trace = []
                st.execute(first)
                trace.append({"phase": "forced_first", "action": first, "observation": st.observations[-1]})
                checkpoints = {}
                for step in range(1, 9):
                    action, dist, dual = policy_action(st, scorer, renderer, full=False)
                    st.execute(action)
                    trace.append({"phase": f"continue_{step}", "action": action, "observation": st.observations[-1], "prompt_snapshot_hash": dual["snapshot_hash"]})
                    if step in (4, 8):
                        checkpoints[str(step)] = endpoint(st, initial)
                branches[label] = {"first_action": first, "trace": trace, "checkpoints": checkpoints}
            row = {
                "schema_version": "joint_recall_fresh_state_v1",
                "component": COMPONENT,
                "seed": args.seed,
                "state_index": selected.index(qid),
                "state_id": f"fresh_seed{args.seed}_{selected.index(qid):04d}",
                "query_id": qid,
                "gold_evidence_ids": sorted(qrels[qid]),
                "snapshot_hash": initial_snapshot.content_hash(),
                "initial_snapshot": initial_snapshot.to_dict(),
                "component_mask_teacher_first": {"importance_tagging": True, "subtractive_curation": True},
                "component_mask_student_first": {"importance_tagging": False, "subtractive_curation": False},
                "continuation_policy": "reduced",
                "full_harness_takeover": False,
                "search_backend": backend,
                "branch_S": branches["S"],
                "branch_T": branches["T"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"shard": args.shard_index, "done": local_index + 1, "total": len(qids), "qid": qid}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
