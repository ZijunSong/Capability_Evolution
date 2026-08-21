#!/usr/bin/env python3
"""Pre-OPD same-state fork for importance_tagging + subtractive_curation.

Contract:
  - same xi_t snapshot for both branches
  - Teacher/Full view has both importance_tagging and subtractive_curation ON
  - Student/Reduced view has both components OFF
  - Teacher is used only for the first fork action
  - both continuations use the reduced policy; no full-harness takeover
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scape.adapters.components import full_mask
from scape.common.hashing import stable_split
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot, capture_snapshot

# Reuse the validated live fork primitives from the prior H100-2 runner.
from run_h100_2_live_fork_replay import (  # noqa: E402
    ARG_THRESHOLD,
    HFContinuationScorer,
    LocalCorpusSearcher,
    _LocalHit,
    _doc_text,
    _mean,
    _stable_float,
    action_distance,
    action_for_tool,
    build_searcher,
    policy_action as single_policy_action,
    recall,
    run_branch as single_run_branch,
)

try:
    from pyserini.search.lucene import LuceneSearcher
except Exception:  # pragma: no cover
    LuceneSearcher = None

TOOL_NAMES = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
)
JOINT_COMPONENT = "importance_tagging_plus_subtractive_curation"
JOINT_OFF = ("importance_tagging", "subtractive_curation")


def joint_student_mask() -> dict[str, bool]:
    mask = full_mask()
    for component in JOINT_OFF:
        mask[component] = False
    return mask


def _load_queries(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[str(parts[0])] = parts[1]
    return out


def _load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out[str(parts[0])].add(str(parts[2]))
    return dict(out)


class JointLiveState:
    def __init__(self, *, qid: str, query: str, gold: set[str], searcher: Any, branch_seed: str = "") -> None:
        self.qid = qid
        self.query = query
        self.gold = set(gold)
        self.searcher = searcher
        self.component = JOINT_COMPONENT
        self.branch_seed = branch_seed
        self.step = 0
        self.documents: list[dict[str, str]] = []
        self.curated_ids: list[str] = []
        self.read_ids: list[str] = []
        self.successful_read_ids: list[str] = []
        self.read_ids_entered_context: list[str] = []
        self.read_ids_retained_at_endpoint: list[str] = []
        self.verified_supported: list[str] = []
        self.verified_unsupported: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.cost = 0
        self._search(query, k=10)
        self.curated_ids = [d["id"] for d in self.documents[:2]]

    def clone(self, suffix: str) -> "JointLiveState":
        new = object.__new__(JointLiveState)
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

    def _search(self, query: str, k: int = 20) -> None:
        hits = self.searcher.search(query, k)
        docs = []
        for h in hits:
            docs.append({"id": str(h.docid), "text": _doc_text(getattr(h, "raw", None) or "")[:1800]})
        self.documents = docs
        self.cost += 1

    def snapshot(self) -> EnvironmentSnapshot:
        docs_by_id = {d["id"]: d for d in self.documents}
        curated_docs = [docs_by_id[i] for i in self.curated_ids if i in docs_by_id]
        graph_edges = [
            {
                "claim": hashlib.sha256(f"{self.query}:{i}".encode()).hexdigest()[:10],
                "doc_id": i,
                "relation": "supports" if i.split("_", 1)[0] in self.gold else "candidate",
            }
            for i in self.curated_ids[:8]
        ]
        return capture_snapshot(
            query_id=self.qid,
            step=self.step,
            harness_mask=joint_student_mask(),
            working_memory={
                "query": self.query,
                "documents": self.documents[:12],
                "curated_docs": curated_docs,
                "curated_ids": list(self.curated_ids),
                "curated_importance": {i: ("high" if i.split("_", 1)[0] in self.gold else "medium") for i in self.curated_ids},
                "evidence_graph": {"nodes": list(self.curated_ids[:8]), "edges": graph_edges},
                "token_budget_marker": f"remaining={max(0, 8192 - self.cost * 256)}",
                "rerank_instruction": "prefer direct evidence and diverse corroborating sources",
                "auto_populate_seed": [self.query],
                "chunk_neighbors": [d["id"] for d in self.documents[1:5]],
                "verified_supported": list(self.verified_supported),
                "verified_unsupported": list(self.verified_unsupported),
            },
            tool_history=self.history,
            observations=self.observations,
            metadata={"backend": "joint_preopd_live_bm25_fork", "branch_seed": self.branch_seed},
        )

    def execute(self, action: Mapping[str, Any]) -> None:
        name = str(action.get("name") or "end_search")
        args = dict(action.get("arguments") or {})
        before_curated = set(self.curated_ids)
        ok = True
        if name in {"search_corpus", "fan_out_search"}:
            if name == "fan_out_search":
                qs = args.get("queries") or [self.query]
                q = str(qs[min(len(qs) - 1, int(_stable_float(self.branch_seed + str(self.step)) * len(qs)))])
            else:
                q = str(args.get("query") or self.query)
            self._search(q, k=20)
        elif name == "grep_corpus":
            pat = str(args.get("pattern") or "").lower()
            filtered = [d for d in self.documents if pat and pat in d.get("text", "").lower()]
            if filtered:
                self.documents = filtered + [d for d in self.documents if d not in filtered]
            self.cost += 1
        elif name == "read_document":
            did = str(args.get("doc_id") or "")
            available = {str(d.get("id")) for d in self.documents}
            if did and did in available:
                if did not in self.read_ids:
                    self.read_ids.append(did)
                if did not in self.successful_read_ids:
                    self.successful_read_ids.append(did)
                if did not in self.read_ids_entered_context:
                    self.read_ids_entered_context.append(did)
                if did not in self.read_ids_retained_at_endpoint:
                    self.read_ids_retained_at_endpoint.append(did)
                ok = True
            else:
                ok = False
            self.cost += 1
        elif name == "review_docs":
            available = {str(d.get("id")) for d in self.documents}
            for did in args.get("doc_ids") or []:
                sid = str(did)
                if sid in available:
                    if sid not in self.read_ids:
                        self.read_ids.append(sid)
                    if sid not in self.successful_read_ids:
                        self.successful_read_ids.append(sid)
                    if sid not in self.read_ids_entered_context:
                        self.read_ids_entered_context.append(sid)
                    if sid not in self.read_ids_retained_at_endpoint:
                        self.read_ids_retained_at_endpoint.append(sid)
                else:
                    ok = False
            self.cost += 1
        elif name == "curate":
            for did in args.get("remove_ids") or []:
                sid = str(did)
                self.curated_ids = [x for x in self.curated_ids if x != sid]
            for did in args.get("add_ids") or []:
                sid = str(did)
                if sid and sid not in self.curated_ids:
                    self.curated_ids.append(sid)
            self.cost += 1
        elif name == "verify":
            for did in args.get("doc_ids") or ([args.get("doc_id")] if args.get("doc_id") else []):
                sid = str(did)
                if sid.split("_", 1)[0] in self.gold:
                    if sid not in self.verified_supported:
                        self.verified_supported.append(sid)
                elif sid:
                    if sid not in self.verified_unsupported:
                        self.verified_unsupported.append(sid)
            self.cost += 1
        elif name == "end_search":
            self.cost += 0
        else:
            ok = False
        self.history.append({"step": self.step, "action": {"name": name, "arguments": args}})
        self.observations.append({"step": self.step + 1, "ok": ok, "curated_delta": len(set(self.curated_ids) - before_curated), "n_curated": len(self.curated_ids)})
        self.step += 1

    def metrics(self) -> dict[str, float]:
        unique_curated = list(dict.fromkeys(self.curated_ids))
        useful = [i for i in unique_curated if i.split("_", 1)[0] in self.gold]
        redundancy = max(0, len(self.curated_ids) - len(unique_curated)) / max(1, len(self.curated_ids))
        coverage = recall(unique_curated, self.gold)
        verified_supported = len(set(self.verified_supported))
        verified_unsupported = len(set(self.verified_unsupported))
        objective = 0.45 * coverage + 0.20 * (len(useful) / max(1, len(self.gold))) + 0.20 * (verified_supported / max(1, len(self.gold))) - 0.05 * redundancy - 0.015 * self.cost - 0.03 * verified_unsupported
        return {
            "curated_evidence_gain": coverage,
            "useful_unique_docs": float(len(set(useful))),
            "redundancy": redundancy,
            "evidence_coverage": coverage,
            "verified_supported_claims": float(verified_supported),
            "unsupported_claims": float(verified_unsupported),
            "tool_search_cost": float(self.cost),
            "objective_utility": objective,
        }


def policy_action(state: JointLiveState, scorer: HFContinuationScorer, renderer: DualViewRenderer, *, full: bool, tie_jitter: str = "") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snap = state.snapshot()
    dual = renderer.render_pair(snap, student_mask=joint_student_mask(), include_null_controls=False)
    view = dual.full_view if full else dual.student_view
    dist = scorer.distribution(view, state)  # type: ignore[arg-type]
    probs = dict(dist["tool_name_probs"])
    if tie_jitter:
        scores = {k: math.log(max(v, 1e-12)) + 1e-4 * (_stable_float(f"{tie_jitter}:{k}:{state.step}") - 0.5) for k, v in probs.items()}
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        dist["decoded"] = action_for_tool(chosen, state, view)  # type: ignore[arg-type]
    return dist["decoded"], dist, dual.to_dict()


def run_branch(start: JointLiveState, first_action: Mapping[str, Any], *, k: int, scorer: HFContinuationScorer, renderer: DualViewRenderer, label: str, replay_jitter: str = "") -> tuple[JointLiveState, list[dict[str, Any]]]:
    st = start.clone(label)
    trace = []
    st.execute(first_action)
    trace.append({"branch": label, "phase": "forced_first", "action": dict(first_action), "metrics": st.metrics()})
    for i in range(k):
        action, dist, dual = policy_action(st, scorer, renderer, full=False, tie_jitter=f"{replay_jitter}:{i}" if replay_jitter else "")
        st.execute(action)
        trace.append({"branch": label, "phase": f"continue_{i+1}", "action": action, "top_prob": max(dist["tool_name_probs"].values()), "snapshot_hash": dual["snapshot_hash"], "metrics": st.metrics()})
    return st, trace


def state_from_snapshot(snap_dict: Mapping[str, Any], query: str, gold: set[str], searcher: Any) -> JointLiveState:
    snap = EnvironmentSnapshot.from_dict(snap_dict)
    st = JointLiveState(qid=snap.query_id, query=query, gold=gold, searcher=searcher, branch_seed=f"fork:{snap.content_hash()}")
    wm = snap.working_memory
    st.step = snap.step
    st.documents = list(wm.get("documents") or [])
    st.curated_ids = list(wm.get("curated_ids") or [])
    st.verified_supported = list(wm.get("verified_supported") or [])
    st.verified_unsupported = list(wm.get("verified_unsupported") or [])
    st.history = list(snap.tool_history or [])
    st.observations = list(snap.observations or [])
    st.cost = len(st.history)
    return st


def freeze_qids(args: argparse.Namespace, queries: dict[str, str], qrels: dict[str, set[str]], out: Path) -> list[str]:
    eligible = sorted(set(queries) & set(qrels))
    selected, _ = stable_split(eligible, seed=args.seed, n_take=args.n_queries_pool)
    man = {"name": "JOINT_IMPORTANCE_SUBTRACTIVE_PREOPD_POOL", "seed": args.seed, "query_ids": selected, "n_query_pool": len(selected)}
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / f"JOINT_POOL_seed{args.seed}.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected


def collect_candidate_states(qids: list[str], queries: dict[str, str], qrels: dict[str, set[str]], searcher: Any, scorer: HFContinuationScorer, renderer: DualViewRenderer, n_states: int) -> list[dict[str, Any]]:
    states = []
    for qid in qids:
        base = JointLiveState(qid=qid, query=queries[qid], gold=qrels.get(qid, set()), searcher=searcher, branch_seed=f"collect:{JOINT_COMPONENT}:{qid}")
        for _ in range(8):
            a_s, d_s, _ = policy_action(base, scorer, renderer, full=False)
            a_t, d_t, _ = policy_action(base, scorer, renderer, full=True)
            div = action_distance(a_s, a_t)
            if a_s.get("name") != a_t.get("name") or div >= ARG_THRESHOLD:
                snap = base.snapshot()
                states.append({
                    "component": JOINT_COMPONENT,
                    "query_id": qid,
                    "turn_id": base.step,
                    "snapshot": snap.to_dict(),
                    "snapshot_hash": snap.content_hash(),
                    "a_S": a_s,
                    "a_T": a_t,
                    "P_tool_reduced": d_s["tool_name_probs"],
                    "P_tool_full": d_t["tool_name_probs"],
                    "divergence": div,
                    "divergence_type": "tool-name" if a_s.get("name") != a_t.get("name") else "args-only",
                })
                if len(states) >= n_states:
                    return states
            base.execute(a_s)
    return states


def run_utility(args: argparse.Namespace) -> int:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = freeze_qids(args, queries, qrels, out)
    searcher, search_backend = build_searcher(args.index_path, args.corpus_path)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    k = args.K
    shard_path = out / "shards" / f"{JOINT_COMPONENT}_K{k}_seed{args.seed}.jsonl"
    status_path = out / "STATUS_LIVE.md"
    states = collect_candidate_states(qids, queries, qrels, searcher, scorer, renderer, args.n_states)
    rows = []
    with shard_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(states):
            start = state_from_snapshot(item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher)
            s_final, s_trace = run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, label="S")
            t_final, t_trace = run_branch(start, item["a_T"], k=k, scorer=scorer, renderer=renderer, label="T")
            sm = s_final.metrics(); tm = t_final.metrics()
            row = {
                "split": "JOINT_IMPORTANCE_SUBTRACTIVE_PREOPD",
                "seed": args.seed,
                "component": JOINT_COMPONENT,
                "components_on_teacher": list(JOINT_OFF),
                "components_off_student": list(JOINT_OFF),
                "K": k,
                "state_id": f"{JOINT_COMPONENT}_K{k}_seed{args.seed}_{idx:04d}",
                "query_id": item["query_id"],
                "turn_id": item["turn_id"],
                "snapshot_hash": item["snapshot_hash"],
                "a_S": item["a_S"],
                "a_T": item["a_T"],
                "P_tool_reduced": item["P_tool_reduced"],
                "P_tool_full": item["P_tool_full"],
                "divergence": item["divergence"],
                "divergence_type": item["divergence_type"],
                "branch_S_metrics": sm,
                "branch_T_metrics": tm,
                "branch_S_endpoint": {
                    "initial_candidate_evidence_ids": [str(d.get("id")) for d in start.documents],
                    "final_candidate_evidence_ids": [str(d.get("id")) for d in s_final.documents],
                    "initial_curated_ids": list(start.curated_ids),
                    "final_curated_ids": list(s_final.curated_ids),
                    "read_attempt_ids_within_k": list(s_final.read_ids),
                    "successful_read_ids_within_k": list(s_final.successful_read_ids),
                    "read_ids_entered_context": list(s_final.read_ids_entered_context),
                    "read_ids_retained_at_endpoint": list(s_final.read_ids_retained_at_endpoint),
                    "final_activated_evidence_ids": sorted(set(s_final.curated_ids) | set(s_final.read_ids_retained_at_endpoint)),
                },
                "branch_T_endpoint": {
                    "initial_candidate_evidence_ids": [str(d.get("id")) for d in start.documents],
                    "final_candidate_evidence_ids": [str(d.get("id")) for d in t_final.documents],
                    "initial_curated_ids": list(start.curated_ids),
                    "final_curated_ids": list(t_final.curated_ids),
                    "read_attempt_ids_within_k": list(t_final.read_ids),
                    "successful_read_ids_within_k": list(t_final.successful_read_ids),
                    "read_ids_entered_context": list(t_final.read_ids_entered_context),
                    "read_ids_retained_at_endpoint": list(t_final.read_ids_retained_at_endpoint),
                    "final_activated_evidence_ids": sorted(set(t_final.curated_ids) | set(t_final.read_ids_retained_at_endpoint)),
                },
                "branch_T_minus_S": tm["objective_utility"] - sm["objective_utility"],
                "curated_evidence_gain": tm["curated_evidence_gain"] - sm["curated_evidence_gain"],
                "useful_unique_docs": tm["useful_unique_docs"] - sm["useful_unique_docs"],
                "redundancy_change": tm["redundancy"] - sm["redundancy"],
                "evidence_coverage": tm["evidence_coverage"] - sm["evidence_coverage"],
                "verified_supported_claim_status": tm["verified_supported_claims"] - sm["verified_supported_claims"],
                "unsupported_claim_status": tm["unsupported_claims"] - sm["unsupported_claims"],
                "tool_search_cost": tm["tool_search_cost"] - sm["tool_search_cost"],
                "full_harness_takeover": False,
                "branch_S_trace": s_trace,
                "branch_T_trace": t_trace,
                "runner": "joint_preopd_same_state_fork_hf_bm25",
                "search_backend": search_backend,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx % 8 == 0:
                write_status_live(status_path, stage="joint_importance_subtractive_preopd_fork", run_id="joint_importance_subtractive_preopd_0820", n_expected=args.n_states, n_finished=idx + 1, errors=[], extra={"K": k, "seed": args.seed, "device": args.device})
    print(json.dumps({"component": JOINT_COMPONENT, "K": k, "seed": args.seed, "n": len(rows), "path": str(shard_path)}, indent=2), flush=True)
    return 0 if len(rows) >= args.n_states else 2


def _norm_doc(doc_id: Any) -> str:
    return str(doc_id).split("_", 1)[0]


def _endpoint_recall(ids: list[Any], gold: set[str]) -> float:
    g = {_norm_doc(x) for x in gold}
    p = {_norm_doc(x) for x in ids}
    return len(p & g) / max(1, len(g))


def _endpoint_precision(ids: list[Any], gold: set[str]) -> float:
    g = {_norm_doc(x) for x in gold}
    p = {_norm_doc(x) for x in ids}
    return len(p & g) / max(1, len(p))


def aggregate(args: argparse.Namespace) -> int:
    out = args.out_dir
    shard_dir = out / "shards"
    util_rows = []
    for seed in args.seeds:
        for k in args.Ks:
            p = shard_dir / f"{JOINT_COMPONENT}_K{k}_seed{seed}.jsonl"
            if not p.exists():
                raise FileNotFoundError(p)
            with p.open(encoding="utf-8") as f:
                util_rows.extend(json.loads(line) for line in f if line.strip())
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    for row in util_rows:
        gold = qrels.get(str(row["query_id"]), set())
        for branch in ("S", "T"):
            ep = row[f"branch_{branch}_endpoint"]
            ep["candidate_recall"] = _endpoint_recall(ep["final_candidate_evidence_ids"], gold)
            ep["candidate_precision"] = _endpoint_precision(ep["final_candidate_evidence_ids"], gold)
            ep["activated_recall"] = _endpoint_recall(ep["final_activated_evidence_ids"], gold)
            ep["activated_precision"] = _endpoint_precision(ep["final_activated_evidence_ids"], gold)
        row["candidate_recall_delta"] = row["branch_T_endpoint"]["candidate_recall"] - row["branch_S_endpoint"]["candidate_recall"]
        row["activated_recall_delta"] = row["branch_T_endpoint"]["activated_recall"] - row["branch_S_endpoint"]["activated_recall"]
    with (out / "JOINT_PREOPD_VALUE_PER_STATE.jsonl").open("w", encoding="utf-8") as f:
        for row in util_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = []
    for seed in args.seeds:
        for k in args.Ks:
            rows = [r for r in util_rows if int(r["seed"]) == seed and int(r["K"]) == k]
            vals = [float(r["branch_T_minus_S"]) for r in rows]
            mean = _mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(len(vals)) if vals else 0.0
            summary.append({
                "component": JOINT_COMPONENT,
                "seed": seed,
                "K": k,
                "n_states": len(rows),
                "mean_branch_T_minus_S": mean,
                "percent_gain": mean * 100.0,
                "median_branch_T_minus_S": statistics.median(vals) if vals else 0.0,
                "ci95_low_normal_approx": mean - ci,
                "ci95_high_normal_approx": mean + ci,
                "positive_count": sum(v > 0 for v in vals),
                "negative_count": sum(v < 0 for v in vals),
                "zero_count": sum(v == 0 for v in vals),
                "mean_curated_evidence_gain": _mean([float(r["curated_evidence_gain"]) for r in rows]),
                "mean_useful_unique_docs": _mean([float(r["useful_unique_docs"]) for r in rows]),
                "mean_evidence_coverage": _mean([float(r["evidence_coverage"]) for r in rows]),
                "mean_tool_search_cost_delta": _mean([float(r["tool_search_cost"]) for r in rows]),
                "candidate_delta_pp": 100.0 * _mean([float(r["candidate_recall_delta"]) for r in rows]),
                "activated_delta_pp": 100.0 * _mean([float(r["activated_recall_delta"]) for r in rows]),
                "candidate_teacher_mean": _mean([float(r["branch_T_endpoint"]["candidate_recall"]) for r in rows]),
                "candidate_student_mean": _mean([float(r["branch_S_endpoint"]["candidate_recall"]) for r in rows]),
                "activated_teacher_mean": _mean([float(r["branch_T_endpoint"]["activated_recall"]) for r in rows]),
                "activated_student_mean": _mean([float(r["branch_S_endpoint"]["activated_recall"]) for r in rows]),
            })
    with (out / "JOINT_PREOPD_SUMMARY.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(summary[0]) if summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary)

    k_means = {}
    for k in args.Ks:
        rows = [r for r in summary if int(r["K"]) == k]
        k_means[str(k)] = _mean([float(r["mean_branch_T_minus_S"]) for r in rows])
    gate_passed = all(v > 0 for v in k_means.values()) and all(int(r["n_states"]) >= args.target_states for r in summary)
    payload = {
        "status": "joint_preopd_k4_k8_gate_passed" if gate_passed else "joint_preopd_k4_k8_gate_failed",
        "component": JOINT_COMPONENT,
        "contract": "same xi_t; Teacher/Full has importance_tagging+subtractive_curation ON for first branch; Student/Reduced has both OFF; both continuations reduced policy; no full-harness takeover",
        "seeds": args.seeds,
        "K": args.Ks,
        "target_states_per_seed_k": args.target_states,
        "n_rows": len(util_rows),
        "mean_by_K": k_means,
        "percent_gain_by_K": {k: v * 100.0 for k, v in k_means.items()},
        "gate_passed": gate_passed,
        "decision": "joint_bundle_preopd_gain_pass" if gate_passed else "do_not_treat_joint_bundle_as_preopd_gain_pass",
        "rows": summary,
    }
    (out / "JOINT_PREOPD_K4_K8_GATE.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# JOINT_PREOPD_K4_K8_GATE",
        "",
        f"- status: `{payload['status']}`",
        f"- component: `{JOINT_COMPONENT}`",
        "- contract: same xi_t; Teacher/Full on for importance_tagging + subtractive_curation; Student/Reduced off for both; reduced continuation; no full takeover",
        "",
        "| seed | K | n | mean T-S | percent gain | CI95 low | CI95 high | pos/neg/zero |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['seed']} | {r['K']} | {r['n_states']} | {r['mean_branch_T_minus_S']:.6f} | {r['percent_gain']:+.2f}% | {r['ci95_low_normal_approx']:.6f} | {r['ci95_high_normal_approx']:.6f} | {r['positive_count']}/{r['negative_count']}/{r['zero_count']} |")
    (out / "JOINT_PREOPD_K4_K8_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = build_run_manifest(run_id="joint_importance_subtractive_preopd_0820", stage="joint_preopd_k4_k8_fork", command=sys.argv, repo_root=REPO, output_dir=out, input_paths={}, extra={"runner": "joint_preopd_same_state_fork_hf_bm25", "K": args.Ks, "seeds": args.seeds, "n_states_per_seed_k": args.target_states})
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=[f"K{k}_seed{s}" for s in args.seeds for k in args.Ks] + ["aggregate"]))
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    write_status_live(out / "STATUS_LIVE.md", stage="joint_preopd_k4_k8_fork", run_id="joint_importance_subtractive_preopd_0820", n_expected=len(args.seeds) * len(args.Ks), n_finished=len(args.seeds) * len(args.Ks), errors=[], extra={"status": payload["status"]})
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["utility", "aggregate"], required=True)
    ap.add_argument("--K", type=int, choices=[4, 8])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--corpus-path", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "0820_joint_importance_subtractive_preopd_fork")
    ap.add_argument("--seed", type=int, default=8423)
    ap.add_argument("--seeds", type=int, nargs="+", default=[8423, 8424])
    ap.add_argument("--Ks", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n-states", type=int, default=512)
    ap.add_argument("--target-states", type=int, default=512)
    ap.add_argument("--n-queries-pool", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    args = ap.parse_args()
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    if args.mode == "utility":
        if not args.K:
            raise SystemExit("--K required for --mode utility")
        return run_utility(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
