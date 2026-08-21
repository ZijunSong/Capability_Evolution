#!/usr/bin/env python3
"""Streaming/batched H100-2 true live fork runner.

This wraps run_h100_2_live_fork_replay.py but replaces collection with streaming
candidate processing and replaces per-tool scorer calls with one batched forward
per policy decision. It is still a true fork/replay runner: a_S/a_T are scored by
HF continuation logprob from the same xi_t, then executable LiveState branches are
forked and continued for K steps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_h100_2_live_fork_replay as base
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.rendering.dual_view import DualViewRenderer


class BatchedHFContinuationScorer(base.HFContinuationScorer):
    def distribution(self, view: Mapping[str, Any], state: base.LiveState) -> dict[str, Any]:
        prompt = base._prompt_for_view(view)
        continuations = [base._call_text(name, state, view) for name in base.TOOL_NAMES]
        scores = self.batch_sequence_logprobs(prompt, continuations)
        score_by_name = dict(zip(base.TOOL_NAMES, scores))
        m = max(score_by_name.values())
        z = m + math.log(sum(math.exp(v - m) for v in score_by_name.values()))
        probs = {k: math.exp(v - z) for k, v in score_by_name.items()}
        decoded = max(probs.items(), key=lambda kv: kv[1])[0]
        return {"tool_name_probs": probs, "decoded": base.action_for_tool(decoded, state, view), "sequence_logprobs": score_by_name}

    def batch_sequence_logprobs(self, prompt: str, continuations: list[str]) -> list[float]:
        torch = self.torch
        pids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(pids) > self.max_prompt_tokens:
            pids = pids[-self.max_prompt_tokens:]
        cid_list = [self.tokenizer.encode(c, add_special_tokens=False) for c in continuations]
        seqs = [pids + cids for cids in cid_list]
        max_len = max(len(s) for s in seqs)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        input_ids = []
        attn = []
        for s in seqs:
            pad = max_len - len(s)
            input_ids.append(s + [pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        mask = torch.tensor(attn, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        out = []
        start = len(pids) - 1
        for b, cids in enumerate(cid_list):
            vals = []
            for j, tok in enumerate(cids):
                vals.append(float(logp[b, start + j, tok].detach().cpu()))
            out.append(float(sum(vals)))
        return out


def candidate_items(component: str, qids: list[str], queries: dict[str, str], qrels: dict[str, set[str]], searcher: Any, scorer: BatchedHFContinuationScorer, renderer: DualViewRenderer):
    for qid in qids:
        st = base.LiveState(qid=qid, query=queries[qid], gold=qrels.get(qid, set()), searcher=searcher, component=component, branch_seed=f"collect:{component}:{qid}")
        for _ in range(8):
            a_s, d_s, _ = base.policy_action(st, scorer, renderer, component=component, full=False)
            a_t, d_t, _ = base.policy_action(st, scorer, renderer, component=component, full=True)
            div = base.action_distance(a_s, a_t)
            if a_s.get("name") != a_t.get("name") or div >= base.ARG_THRESHOLD:
                snap = st.snapshot()
                yield {
                    "component": component,
                    "query_id": qid,
                    "turn_id": st.step,
                    "snapshot": snap.to_dict(),
                    "snapshot_hash": snap.content_hash(),
                    "a_S": a_s,
                    "a_T": a_t,
                    "P_tool_reduced": d_s["tool_name_probs"],
                    "P_tool_full": d_t["tool_name_probs"],
                    "divergence": div,
                    "divergence_type": "tool-name" if a_s.get("name") != a_t.get("name") else "args-only",
                }
            st.execute(a_s)


def row_for_item(args: argparse.Namespace, item: dict[str, Any], queries: dict[str, str], qrels: dict[str, set[str]], searcher: Any, scorer: BatchedHFContinuationScorer, renderer: DualViewRenderer, idx: int) -> dict[str, Any]:
    component = args.component
    k = args.K
    start = base.state_from_snapshot(item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher, component)
    initial_candidate_ids = [str(d["id"]) for d in start.documents]
    initial_curated_ids = list(start.curated_ids)
    s_final, s_trace = base.run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, component=component, label="S")
    t_final, t_trace = base.run_branch(start, item["a_T"], k=k, scorer=scorer, renderer=renderer, component=component, label="T")
    sm = s_final.metrics(); tm = t_final.metrics()

    def endpoint(st: base.LiveState, trace: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = [str(x.get("action", {}).get("arguments", {}).get("doc_id"))
                    for x in trace if x.get("action", {}).get("name") == "read_document"
                    and x.get("action", {}).get("arguments", {}).get("doc_id")]
        # LiveState.execute records successful read observations for every valid
        # read_document action; retained IDs are the context-visible read IDs at K.
        successful = list(dict.fromkeys(st.read_ids))
        return {
            "initial_candidate_evidence_ids": list(initial_candidate_ids),
            "final_candidate_evidence_ids": [str(d["id"]) for d in st.documents],
            "initial_curated_ids": list(initial_curated_ids),
            "final_curated_ids": list(st.curated_ids),
            "read_attempt_ids_within_k": attempts,
            "successful_read_ids_within_k": successful,
            "read_ids_entered_context": list(successful),
            "read_ids_retained_at_endpoint": list(successful),
            "final_activated_evidence_ids": list(dict.fromkeys(st.curated_ids + successful)),
            "actions": [x.get("action", {}) for x in trace],
            "observations": list(st.observations),
            "context_evidence_ids_by_step": [list(successful)],
            "initial_state_hash": item["snapshot_hash"],
            "final_state_hash": st.snapshot().content_hash(),
            "tool_cost": float(st.cost),
            "duplicate_read_count": max(0, len(attempts) - len(set(attempts))),
        }

    ep_s = endpoint(s_final, s_trace)
    ep_t = endpoint(t_final, t_trace)
    return {
        "split": "UTILITY_LIVE256",
        "seed": args.seed,
        "component": component,
        "K": k,
        "state_id": f"{component}_K{k}_{idx:03d}",
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
        "branch_S_endpoint": ep_s,
        "branch_T_endpoint": ep_t,
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
        "runner": "true_live_fork_replay_hf_bm25_batched_stream",
    }


def run_utility(args: argparse.Namespace) -> int:
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True); (out / "shards").mkdir(exist_ok=True)
    queries = base._load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = base.freeze_qids(args, queries, qrels, out)
    searcher, _ = base.build_searcher(args.index_path, args.corpus_path)
    scorer = BatchedHFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    path = out / "shards" / f"{args.component}_K{args.K}.jsonl"
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in candidate_items(args.component, qids, queries, qrels, searcher, scorer, renderer):
            row = row_for_item(args, item, queries, qrels, searcher, scorer, renderer, n)
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 4 == 0:
                write_status_live(out / f"STATUS_{args.component}_K{args.K}.md", stage="h100_2_candidate_b_live_utility", run_id="h1002_true_live_stream", n_expected=args.n_states, n_finished=n, errors=[], extra={"component": args.component, "K": args.K, "runner": "true_live_fork_replay_hf_bm25_batched_stream"})
                print(json.dumps({"component": args.component, "K": args.K, "n": n}), flush=True)
            if n >= args.n_states:
                break
    print(json.dumps({"component": args.component, "K": args.K, "n": n, "path": str(path)}), flush=True)
    return 0 if n >= args.n_states else 2


def run_noise(args: argparse.Namespace) -> int:
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True); (out / "shards").mkdir(exist_ok=True)
    queries = base._load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = base.freeze_qids(args, queries, qrels, out)
    searcher, _ = base.build_searcher(args.index_path, args.corpus_path)
    scorer = BatchedHFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    path = out / "shards" / "replay_noise.jsonl"
    n_total = 0
    target_each = max(1, args.n_states // len(base.COMPONENTS))
    with path.open("w", encoding="utf-8") as f:
        for component in base.COMPONENTS:
            n = 0
            for item in candidate_items(component, qids, queries, qrels, searcher, scorer, renderer):
                for k in (2, 4):
                    start = base.state_from_snapshot(item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher, component)
                    n1, tr1 = base.run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, component=component, label="N1", replay_jitter="N1")
                    n2, tr2 = base.run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, component=component, label="N2", replay_jitter="N2")
                    m1 = n1.metrics(); m2 = n2.metrics()
                    row = {"split": "UTILITY_LIVE256", "seed": args.seed, "component": component, "K": k, "state_id": f"noise_{component}_K{k}_{n:03d}", "query_id": item["query_id"], "snapshot_hash": item["snapshot_hash"], "a_S": item["a_S"], "branch_N1_metrics": m1, "branch_N2_metrics": m2, "replay_noise": abs(m1["objective_utility"] - m2["objective_utility"]), "branch_N1_trace": tr1, "branch_N2_trace": tr2, "runner": "true_live_same_action_replay_hf_bm25_batched_stream"}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); n_total += 1
                n += 1
                if n % 4 == 0:
                    print(json.dumps({"component": component, "states": n, "noise_rows": n_total}), flush=True)
                if n >= target_each:
                    break
    print(json.dumps({"noise_rows": n_total, "path": str(path)}), flush=True)
    return 0


def aggregate(args: argparse.Namespace) -> int:
    rc = base.aggregate(args)
    # Patch runner labels in decision/manifest markdown to show streamed batched runner.
    out = args.out_dir
    for p in [out / "CANDIDATE_B_LIVE_DECISION.md", out / "SUBTRACTIVE_LIVE_UTILITY.md", out / "IMPORTANCE_LIVE_UTILITY.md", out / "VERIFY_LIVE_UTILITY.md"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8").replace("true_live_fork_replay_hf_bm25", "true_live_fork_replay_hf_bm25_batched_stream")
            p.write_text(txt, encoding="utf-8")
    if (out / "CANDIDATE_B_LIVE_DECISION.json").exists():
        obj = json.loads((out / "CANDIDATE_B_LIVE_DECISION.json").read_text(encoding="utf-8")); obj["runner"] = "true_live_fork_replay_hf_bm25_batched_stream"
        (out / "CANDIDATE_B_LIVE_DECISION.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pre = out.parent / "scape_prestage_v3"; pre.mkdir(parents=True, exist_ok=True)
        (pre / "H1002_CANDIDATE_B_LIVE_HANDOFF.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    return rc


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["utility", "noise", "aggregate"], required=True)
    ap.add_argument("--component", choices=base.COMPONENTS)
    ap.add_argument("--K", type=int, choices=[2, 4, 8])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="/mnt/songzijun/models/harness-1")
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--corpus-path", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_2_candidate_b_live_utility")
    ap.add_argument("--seed", type=int, default=2214)
    ap.add_argument("--n-states", type=int, default=256)
    ap.add_argument("--n-queries-pool", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = ap.parse_args()
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    if args.mode == "utility":
        if not args.component or args.K is None: raise SystemExit("--component and --K required")
        return run_utility(args)
    if args.mode == "noise":
        return run_noise(args)
    return aggregate(args)

if __name__ == "__main__":
    raise SystemExit(main())
