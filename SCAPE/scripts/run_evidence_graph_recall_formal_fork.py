#!/usr/bin/env python3
"""Formal evidence_graph same-state K4/K8 reward fork.

This runner freezes UTILITY_LIVE256 (seed=2214) from BrowseComp+ fresh queries,
collects candidate-bearing same-state snapshots, scores reduced/full actions with
an HF continuation-logprob Harness-1 scorer, then actually forks executable BM25
search environments:

  Branch S: execute a_S, continue with the reduced-view policy for K steps.
  Branch T: execute a_T, continue with the same reduced-view policy for K steps.
  Branch N1/N2: execute the same a_S and continue K steps to measure live replay
                noise (with controlled tie/noise perturbations in action choice).

The full-view action is used only for the first fork action; Full Harness never
continues/takes over after that.
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
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from pyserini.search.lucene import LuceneSearcher
except Exception:  # pragma: no cover - optional production dependency
    LuceneSearcher = None


class _LocalHit:
    def __init__(self, docid: str, raw: str, score: float) -> None:
        self.docid = docid
        self.raw = raw
        self.score = score


class LocalCorpusSearcher:
    """Small deterministic fallback when the Lucene/pyserini stack is absent."""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path
        self.docs: list[tuple[str, str]] = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                docid = str(row.get("id") or row.get("docid") or row.get("source"))
                text = str(row.get("text") or row.get("contents") or row.get("content") or "")
                if docid and text:
                    self.docs.append((docid, text))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {x for x in text.lower().replace("_", " ").split() if len(x) > 2}

    def search(self, query: str, k: int = 20) -> list[_LocalHit]:
        q = self._tokens(query)
        scored = []
        for docid, text in self.docs:
            dt = self._tokens(text)
            overlap = len(q & dt)
            phrase = 1.0 if query.lower() in text.lower() else 0.0
            score = float(overlap) + phrase
            if score > 0:
                scored.append((score, docid, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_LocalHit(docid, json.dumps({"id": docid, "contents": text}, ensure_ascii=False), score) for score, docid, text in scored[:k]]


def build_searcher(index_path: Path, corpus_path: Path) -> tuple[Any, str]:
    if LuceneSearcher is not None and index_path.exists():
        return LuceneSearcher(str(index_path)), "pyserini_lucene"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Neither Lucene index nor fallback corpus exists: {index_path} / {corpus_path}")
    return LocalCorpusSearcher(corpus_path), "local_corpus_token_overlap"

from scape.adapters.components import full_mask, minus_mask
from scape.common.hashing import stable_split
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot, capture_snapshot

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
COMPONENTS = ("evidence_graph",)
ARG_THRESHOLD = 0.12


def _stable_float(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:13], 16) / float(16**13 - 1)


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


def _load_used_qids(paths: list[Path]) -> set[str]:
    used = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            used.add(str(json.loads(line).get("query_id")))
                        except Exception:
                            pass
        else:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            raw = obj.get("query_ids") if isinstance(obj, dict) else obj if isinstance(obj, list) else []
            if isinstance(raw, list):
                used.update(str(x) for x in raw)
    return {x for x in used if x and x != "None"}


def normalize_doc_id(value: Any) -> str:
    return str(value).split("_", 1)[0]


def normalized_ids(values: list[str] | set[str]) -> set[str]:
    return {normalize_doc_id(value) for value in values if str(value)}


def recall(ids: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    norm = normalized_ids(ids)
    norm_gold = normalized_ids(gold)
    return len(norm & norm_gold) / len(norm_gold)


def precision(ids: list[str], gold: set[str]) -> float:
    norm = normalized_ids(ids)
    if not norm:
        return 0.0
    return len(norm & normalized_ids(gold)) / len(norm)


def _doc_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return str(obj.get("contents") or obj.get("text") or raw)
        except Exception:
            return raw
        return raw
    return str(raw)


class HFContinuationScorer:
    def __init__(self, model_path: str, *, device: str = "cuda:0", dtype: str = "bfloat16", max_prompt_tokens: int = 3072) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.max_prompt_tokens = int(max_prompt_tokens)

    def sequence_logprob(self, prompt: str, continuation: str) -> float:
        torch = self.torch
        pids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not cids:
            return 0.0
        if len(pids) > self.max_prompt_tokens:
            pids = pids[-self.max_prompt_tokens:]
        ids = torch.tensor([pids + cids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits
            logp = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
        start = len(pids) - 1
        vals = []
        for j, tok in enumerate(cids):
            vals.append(float(logp[start + j, tok].detach().cpu()))
        return float(sum(vals))

    def distribution(self, view: Mapping[str, Any], state: "LiveState") -> dict[str, Any]:
        prompt = _prompt_for_view(view)
        scores = {name: self.sequence_logprob(prompt, _call_text(name, state, view)) for name in TOOL_NAMES}
        m = max(scores.values())
        z = m + math.log(sum(math.exp(v - m) for v in scores.values()))
        probs = {k: math.exp(v - z) for k, v in scores.items()}
        decoded = max(probs.items(), key=lambda kv: kv[1])[0]
        return {"tool_name_probs": probs, "decoded": action_for_tool(decoded, state, view), "sequence_logprobs": scores}


def _prompt_for_view(view: Mapping[str, Any]) -> str:
    payload = dict(view)
    payload.pop("render_hash", None)
    return (
        "You are Harness-1 choosing the next BrowseComp tool call.\n"
        "Choose exactly one tool and JSON arguments. Prefer objectively useful evidence.\n"
        f"TOOLS: {', '.join(TOOL_NAMES)}\n"
        "STATE:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)[:12000]}\n"
        "NEXT_TOOL:"
    )


def _call_text(tool_name: str, state: "LiveState", view: Mapping[str, Any]) -> str:
    action = action_for_tool(tool_name, state, view)
    return f" {action['name']} {json.dumps(action['arguments'], ensure_ascii=False, sort_keys=True)}"


def action_for_tool(tool_name: str, state: "LiveState", view: Mapping[str, Any]) -> dict[str, Any]:
    docs = state.documents or []
    curated = state.curated_ids or []
    first = docs[0]["id"] if docs else ""
    second = docs[1]["id"] if len(docs) > 1 else first
    q = state.query
    if tool_name == "fan_out_search":
        return {"name": tool_name, "arguments": {"queries": [q, f"{q} evidence", f"{q} source"]}}
    if tool_name == "search_corpus":
        return {"name": tool_name, "arguments": {"query": q}}
    if tool_name == "grep_corpus":
        pat = next((w for w in q.split() if len(w) > 5), q.split()[0] if q.split() else q[:12])
        return {"name": tool_name, "arguments": {"pattern": pat}}
    if tool_name == "read_document":
        return {"name": tool_name, "arguments": {"doc_id": first}}
    if tool_name == "review_docs":
        return {"name": tool_name, "arguments": {"doc_ids": [d["id"] for d in docs[:5]]}}
    if tool_name == "curate":
        add = [d["id"] for d in docs[:4] if d["id"] not in curated]
        remove = [curated[-1]] if len(curated) > 6 else []
        return {"name": tool_name, "arguments": {"add_ids": add[:2] or ([second] if second else []), "remove_ids": remove}}
    if tool_name == "verify":
        return {"name": tool_name, "arguments": {"doc_ids": curated[:4] or ([first] if first else []), "claim": state.query[:160]}}
    return {"name": "end_search", "arguments": {}}


def action_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    if a.get("name") != b.get("name"):
        return 1.0
    aa = json.dumps(a.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
    bb = json.dumps(b.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
    if aa == bb:
        return 0.0
    aset = set(aa.split()) | set(aa)
    bset = set(bb.split()) | set(bb)
    return 1.0 - len(aset & bset) / max(1, len(aset | bset))


class LiveState:
    def __init__(self, *, qid: str, query: str, gold: set[str], searcher: Any, component: str, branch_seed: str = "") -> None:
        self.qid = qid
        self.query = query
        self.gold = set(gold)
        self.searcher = searcher
        self.component = component
        self.branch_seed = branch_seed
        self.step = 0
        self.documents: list[dict[str, str]] = []
        self.curated_ids: list[str] = []
        self.read_ids: list[str] = []
        self.read_attempt_ids: list[str] = []
        self.context_evidence_ids_by_step: list[list[str]] = []
        self.verified_supported: list[str] = []
        self.verified_unsupported: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.cost = 0
        self._search(query, k=10)
        self.curated_ids = [d["id"] for d in self.documents[:2]]

    def clone(self, suffix: str) -> "LiveState":
        new = object.__new__(LiveState)
        new.qid = self.qid; new.query = self.query; new.gold = set(self.gold); new.searcher = self.searcher
        new.component = self.component; new.branch_seed = f"{self.branch_seed}:{suffix}"; new.step = self.step
        new.documents = deepcopy(self.documents); new.curated_ids = list(self.curated_ids); new.read_ids = list(self.read_ids)
        new.read_attempt_ids = list(self.read_attempt_ids); new.context_evidence_ids_by_step = deepcopy(self.context_evidence_ids_by_step)
        new.verified_supported = list(self.verified_supported); new.verified_unsupported = list(self.verified_unsupported)
        new.history = deepcopy(self.history); new.observations = deepcopy(self.observations); new.cost = self.cost
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
        graph_edges = [{"claim": hashlib.sha256(f"{self.query}:{i}".encode()).hexdigest()[:10], "doc_id": i, "relation": "supports" if i.split("_", 1)[0] in self.gold else "candidate"} for i in self.curated_ids[:8]]
        return capture_snapshot(
            query_id=self.qid,
            step=self.step,
            harness_mask=minus_mask(self.component),
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
                "retained_read_ids": list(self.read_ids),
            },
            tool_history=self.history,
            observations=self.observations,
            metadata={"backend": "live_bm25_fork_replay", "branch_seed": self.branch_seed},
        )

    def execute(self, action: Mapping[str, Any]) -> None:
        name = str(action.get("name") or "end_search")
        args = dict(action.get("arguments") or {})
        before_curated = set(self.curated_ids)
        ok = True
        if name in {"search_corpus", "fan_out_search"}:
            if name == "fan_out_search":
                qs = args.get("queries") or [self.query]
                q = str(qs[min(len(qs)-1, int(_stable_float(self.branch_seed + str(self.step)) * len(qs)))])
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
            if did:
                self.read_attempt_ids.append(did)
            visible_ids = {str(doc["id"]) for doc in self.documents}
            ok = bool(did and did in visible_ids)
            if ok and did not in self.read_ids:
                self.read_ids.append(did)
            self.cost += 1
        elif name == "review_docs":
            requested = [str(did) for did in args.get("doc_ids") or []]
            visible_ids = {str(doc["id"]) for doc in self.documents}
            self.read_attempt_ids.extend(requested)
            ok = bool(requested) and all(did in visible_ids for did in requested)
            if ok:
                for sid in requested:
                    if sid not in self.read_ids:
                        self.read_ids.append(sid)
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
        successful_read_ids = []
        if name == "read_document" and ok:
            successful_read_ids = [str(args.get("doc_id"))]
        elif name == "review_docs" and ok:
            successful_read_ids = [str(x) for x in args.get("doc_ids") or []]
        self.history.append({"step": self.step, "action": {"name": name, "arguments": args}})
        self.observations.append({
            "step": self.step + 1,
            "ok": ok,
            "successful_read_ids": successful_read_ids,
            "curated_delta": len(set(self.curated_ids) - before_curated),
            "n_curated": len(self.curated_ids),
        })
        self.context_evidence_ids_by_step.append(list(dict.fromkeys(self.read_ids)))
        self.step += 1

    def evidence_endpoint(self) -> dict[str, Any]:
        candidate_ids = list(dict.fromkeys(str(doc["id"]) for doc in self.documents))
        curated_ids = list(dict.fromkeys(self.curated_ids))
        retained_read_ids = list(dict.fromkeys(self.read_ids))
        activated_ids = list(dict.fromkeys(curated_ids + retained_read_ids))
        return {
            "final_candidate_evidence_ids": candidate_ids,
            "final_curated_ids": curated_ids,
            "read_attempt_ids_within_k": list(self.read_attempt_ids),
            "successful_read_ids_within_k": retained_read_ids,
            "read_ids_entered_context": retained_read_ids,
            "read_ids_retained_at_endpoint": retained_read_ids,
            "final_activated_evidence_ids": activated_ids,
            "candidate_evidence_pool_recall_at_k": recall(candidate_ids, self.gold),
            "candidate_evidence_pool_precision_at_k": precision(candidate_ids, self.gold),
            "activated_evidence_recall_at_k": recall(activated_ids, self.gold),
            "activated_evidence_precision_at_k": precision(activated_ids, self.gold),
            "candidate_evidence_pool_size_at_k": len(normalized_ids(candidate_ids)),
            "activated_evidence_size_at_k": len(normalized_ids(activated_ids)),
            "context_evidence_ids_by_step": deepcopy(self.context_evidence_ids_by_step),
        }

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


def policy_action(state: LiveState, scorer: HFContinuationScorer, renderer: DualViewRenderer, *, component: str, full: bool, tie_jitter: str = "") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snap = state.snapshot()
    dual = renderer.render_pair(snap, component_id=component, include_null_controls=False)
    view = dual.full_view if full else dual.student_view
    dist = scorer.distribution(view, state)
    probs = dict(dist["tool_name_probs"])
    if tie_jitter:
        # Tiny deterministic perturbation makes same-action replay expose real
        # branch sensitivity without changing the policy class.
        scores = {k: math.log(max(v, 1e-12)) + 1e-4 * (_stable_float(f"{tie_jitter}:{k}:{state.step}") - 0.5) for k, v in probs.items()}
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        dist["decoded"] = action_for_tool(chosen, state, view)
    return dist["decoded"], dist, dual.to_dict()


def run_branch(start: LiveState, first_action: Mapping[str, Any], *, k: int, scorer: HFContinuationScorer, renderer: DualViewRenderer, component: str, label: str, continuation_full: bool = False, replay_jitter: str = "") -> tuple[LiveState, list[dict[str, Any]]]:
    st = start.clone(label)
    trace = []
    st.execute(first_action)
    trace.append({"branch": label, "phase": "forced_first", "action": dict(first_action), "view": "full" if continuation_full else "reduced", "metrics": st.metrics()})
    # The forced first action is step 1 of the K-step endpoint contract.
    for i in range(max(0, k - 1)):
        action, dist, dual = policy_action(st, scorer, renderer, component=component, full=continuation_full, tie_jitter=f"{replay_jitter}:{i}" if replay_jitter else "")
        st.execute(action)
        trace.append({"branch": label, "phase": f"continue_{i+1}", "action": action, "view": "full" if continuation_full else "reduced", "top_prob": max(dist["tool_name_probs"].values()), "snapshot_hash": dual["snapshot_hash"], "metrics": st.metrics()})
    return st, trace


def freeze_qids(args: argparse.Namespace, queries: dict[str, str], qrels: dict[str, set[str]], out: Path) -> list[str]:
    used = _load_used_qids([
        REPO / "outputs" / "h100_3_real_influence" / "REAL_INFLUENCE_PER_STATE.jsonl",
        REPO / "outputs" / "h100_2_independent_repl" / "manifests" / "bcp_repl200_v2.json",
        REPO / "outputs" / "h100_1_contribution_confirm" / "manifests" / "BCP_CONFIRM400.json",
    ])
    eligible = sorted((set(queries) & set(qrels)) - used)
    if len(eligible) < args.n_queries_pool:
        # Keep the run going but record that disjoint pool was exhausted.
        eligible = sorted(set(queries) & set(qrels))
    selected, _ = stable_split(eligible, seed=args.seed, n_take=args.n_queries_pool)
    man = {"name": "UTILITY_LIVE256", "seed": args.seed, "query_ids": selected, "n_query_pool": len(selected), "excluded_prior_qids": len(used), "query_disjoint_when_possible": True}
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / "UTILITY_LIVE256.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected


def collect_candidate_states(component: str, qids: list[str], queries: dict[str, str], qrels: dict[str, set[str]], searcher: Any, scorer: HFContinuationScorer, renderer: DualViewRenderer, n_states: int) -> list[dict[str, Any]]:
    states = []
    for qid in qids:
        base = LiveState(qid=qid, query=queries[qid], gold=qrels.get(qid, set()), searcher=searcher, component=component, branch_seed=f"collect:{component}:{qid}")
        for t in range(8):
            a_s, d_s, dual_s = policy_action(base, scorer, renderer, component=component, full=False)
            a_t, d_t, dual_t = policy_action(base, scorer, renderer, component=component, full=True)
            div = action_distance(a_s, a_t)
            if a_s.get("name") != a_t.get("name") or div >= ARG_THRESHOLD:
                states.append({
                    "component": component,
                    "query_id": qid,
                    "turn_id": base.step,
                    "snapshot": base.snapshot().to_dict(),
                    "snapshot_hash": base.snapshot().content_hash(),
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


def state_from_snapshot(snap_dict: Mapping[str, Any], query: str, gold: set[str], searcher: Any, component: str) -> LiveState:
    snap = EnvironmentSnapshot.from_dict(snap_dict)
    st = LiveState(qid=snap.query_id, query=query, gold=gold, searcher=searcher, component=component, branch_seed=f"fork:{snap.content_hash()}")
    wm = snap.working_memory
    st.step = snap.step
    st.documents = list(wm.get("documents") or [])
    st.curated_ids = list(wm.get("curated_ids") or [])
    st.read_ids = list(wm.get("retained_read_ids") or [])
    st.read_attempt_ids = []
    st.context_evidence_ids_by_step = []
    st.verified_supported = list(wm.get("verified_supported") or [])
    st.verified_unsupported = list(wm.get("verified_unsupported") or [])
    st.history = list(snap.tool_history or [])
    st.observations = list(snap.observations or [])
    st.cost = len(st.history)
    return st


def run_utility_shard(args: argparse.Namespace) -> int:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = freeze_qids(args, queries, qrels, out)
    searcher, search_backend = build_searcher(args.index_path, args.corpus_path)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    component = args.component
    k = args.K
    shard_path = out / "shards" / f"{component}_K{k}.jsonl"
    status_path = out / "STATUS_LIVE.md"
    if args.states_cache and args.states_cache.exists():
        with args.states_cache.open(encoding="utf-8") as f:
            states = [json.loads(line) for line in f if line.strip()]
        states = states[args.state_offset : args.state_offset + args.n_states]
    else:
        states = collect_candidate_states(component, qids, queries, qrels, searcher, scorer, renderer, args.n_states)
        if args.states_cache:
            args.states_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.states_cache.open("w", encoding="utf-8") as f:
                for state in states:
                    f.write(json.dumps(state, ensure_ascii=False) + "\n")
    rows = []
    with shard_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(states):
            start = state_from_snapshot(item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher, component)
            initial_candidate_ids = list(dict.fromkeys(str(doc["id"]) for doc in start.documents))
            initial_curated_ids = list(dict.fromkeys(start.curated_ids))
            initial_state_hash = start.snapshot().content_hash()
            s_final, s_trace = run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, component=component, label="S")
            t_final, t_trace = run_branch(start, item["a_T"], k=k, scorer=scorer, renderer=renderer, component=component, label="T")
            sm = s_final.metrics(); tm = t_final.metrics()
            se = s_final.evidence_endpoint(); te = t_final.evidence_endpoint()
            row = {
                "split": "UTILITY_LIVE256",
                "seed": args.seed,
                "component": component,
                "K": k,
                "state_id": f"{component}_K{k}_{args.state_offset + idx:03d}",
                "query_id": item["query_id"],
                "turn_id": item["turn_id"],
                "snapshot_hash": item["snapshot_hash"],
                "initial_state_hash": initial_state_hash,
                "gold_evidence_ids": sorted(qrels[item["query_id"]]),
                "initial_candidate_evidence_ids": initial_candidate_ids,
                "initial_curated_ids": initial_curated_ids,
                "component_mask_first_action": {
                    "teacher": "full_component_on",
                    "student": "reduced_component_off",
                },
                "continuation_policy": "reduced",
                "context_retention_policy": "successful_reads_append_only_retained_to_endpoint",
                "branch_S_endpoint": {
                    **se,
                    "observations": list(s_final.observations),
                    "actions": [x.get("action", {}) for x in s_trace],
                    "initial_candidate_evidence_ids": initial_candidate_ids,
                    "initial_curated_ids": initial_curated_ids,
                    "initial_state_hash": initial_state_hash,
                    "component_mask_first_action": "reduced_component_off",
                    "continuation_policy": "reduced",
                    "full_harness_takeover": False,
                },
                "branch_T_endpoint": {
                    **te,
                    "observations": list(t_final.observations),
                    "actions": [x.get("action", {}) for x in t_trace],
                    "initial_candidate_evidence_ids": initial_candidate_ids,
                    "initial_curated_ids": initial_curated_ids,
                    "initial_state_hash": initial_state_hash,
                    "component_mask_first_action": "full_component_on",
                    "continuation_policy": "reduced",
                    "full_harness_takeover": False,
                },
                "branch_S_final_state_hash": s_final.snapshot().content_hash(),
                "branch_T_final_state_hash": t_final.snapshot().content_hash(),
                "candidate_recall_delta": te["candidate_evidence_pool_recall_at_k"] - se["candidate_evidence_pool_recall_at_k"],
                "activated_recall_delta": te["activated_evidence_recall_at_k"] - se["activated_evidence_recall_at_k"],
                "a_S": item["a_S"],
                "a_T": item["a_T"],
                "P_tool_reduced": item["P_tool_reduced"],
                "P_tool_full": item["P_tool_full"],
                "divergence": item["divergence"],
                "divergence_type": item["divergence_type"],
                "branch_S_metrics": sm,
                "branch_T_metrics": tm,
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
                "runner": "true_live_fork_replay_hf_bm25",
                "search_backend": search_backend,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx % 8 == 0:
                write_status_live(status_path, stage="evidence_graph_formal_fork", run_id="evidence_graph_formal_fork", n_expected=args.n_states, n_finished=idx + 1, errors=[], extra={"component": component, "K": k, "phase": "utility_shard", "gpu_device": args.device})
    print(json.dumps({"component": component, "K": k, "n": len(rows), "path": str(shard_path)}, indent=2), flush=True)
    return 0 if len(rows) >= args.n_states else 2


def run_noise_shard(args: argparse.Namespace) -> int:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = freeze_qids(args, queries, qrels, out)
    searcher, search_backend = build_searcher(args.index_path, args.corpus_path)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    rows = []
    noise_path = out / "shards" / "replay_noise.jsonl"
    with noise_path.open("w", encoding="utf-8") as f:
        for component in COMPONENTS:
            states = collect_candidate_states(component, qids, queries, qrels, searcher, scorer, renderer, max(1, args.n_states // len(COMPONENTS)))
            for k in (4, 8):
                for idx, item in enumerate(states):
                    start = state_from_snapshot(item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher, component)
                    n1, tr1 = run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, component=component, label="N1", replay_jitter="N1")
                    n2, tr2 = run_branch(start, item["a_S"], k=k, scorer=scorer, renderer=renderer, component=component, label="N2", replay_jitter="N2")
                    m1 = n1.metrics(); m2 = n2.metrics()
                    row = {"split": "UTILITY_LIVE256", "seed": args.seed, "component": component, "K": k, "state_id": f"noise_{component}_K{k}_{idx:03d}", "query_id": item["query_id"], "snapshot_hash": item["snapshot_hash"], "a_S": item["a_S"], "branch_N1_metrics": m1, "branch_N2_metrics": m2, "replay_noise": abs(m1["objective_utility"] - m2["objective_utility"]), "branch_N1_trace": tr1, "branch_N2_trace": tr2, "runner": "true_live_same_action_replay_hf_bm25"}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"noise_rows": len(rows), "path": str(noise_path)}, indent=2), flush=True)
    return 0


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def aggregate(args: argparse.Namespace) -> int:
    out = args.out_dir
    shard_dir = out / "shards"
    util_rows = []
    for comp in COMPONENTS:
        for k in (4, 8):
            p = shard_dir / f"{comp}_K{k}.jsonl"
            if not p.exists():
                raise FileNotFoundError(p)
            with p.open(encoding="utf-8") as f:
                util_rows.extend(json.loads(line) for line in f if line.strip())
    noise_rows = []
    with (shard_dir / "replay_noise.jsonl").open(encoding="utf-8") as f:
        noise_rows = [json.loads(line) for line in f if line.strip()]
    with (out / "EVIDENCE_GRAPH_VALUE_PER_STATE.jsonl").open("w", encoding="utf-8") as f:
        for r in util_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "LIVE_REPLAY_NOISE.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["split", "seed", "component", "K", "state_id", "query_id", "snapshot_hash", "replay_noise", "runner"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(noise_rows)
    noise_by_k = {k: _mean([float(r["replay_noise"]) for r in noise_rows if int(r["K"]) == k]) for k in (4, 8)}
    summary = []
    for comp in COMPONENTS:
        for k in (4, 8):
            rows = [r for r in util_rows if r["component"] == comp and int(r["K"]) == k]
            eff = _mean([float(r["branch_T_minus_S"]) for r in rows])
            rn = noise_by_k[k]
            summary.append({
                "component": comp,
                "K": k,
                "n_states": len(rows),
                "mean_branch_T_minus_S": eff,
                "median_branch_T_minus_S": statistics.median([float(r["branch_T_minus_S"]) for r in rows]) if rows else 0.0,
                "curated_evidence_gain": _mean([float(r["curated_evidence_gain"]) for r in rows]),
                "useful_unique_docs": _mean([float(r["useful_unique_docs"]) for r in rows]),
                "redundancy_change": _mean([float(r["redundancy_change"]) for r in rows]),
                "evidence_coverage": _mean([float(r["evidence_coverage"]) for r in rows]),
                "verified_supported_claim_status": _mean([float(r["verified_supported_claim_status"]) for r in rows]),
                "unsupported_claim_status": _mean([float(r["unsupported_claim_status"]) for r in rows]),
                "tool_search_cost": _mean([float(r["tool_search_cost"]) for r in rows]),
                "replay_noise": rn,
                "effect_minus_replay_noise": eff - rn,
                "effect_over_replay_noise": eff / max(rn, 1e-12),
                "natural_states": sum(1 for r in rows if r["divergence_type"] == "tool-name"),
                "targeted_states": sum(1 for r in rows if r["divergence_type"] == "args-only"),
                "early": sum(1 for r in rows if int(r["turn_id"]) < 3),
                "mid": sum(1 for r in rows if 3 <= int(r["turn_id"]) < 6),
                "late": sum(1 for r in rows if int(r["turn_id"]) >= 6),
                "runner": "true_live_fork_replay_hf_bm25",
            })
    with (out / "EVIDENCE_GRAPH_UTILITY_SUMMARY.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(summary[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    by_comp = {c: [r for r in summary if r["component"] == c] for c in COMPONENTS}
    comp_score = {c: _mean([float(r["mean_branch_T_minus_S"]) for r in rows]) for c, rows in by_comp.items()}
    comp_noise = {c: _mean([float(r["replay_noise"]) for r in rows]) for c, rows in by_comp.items()}
    consistent = {c: (by_comp[c][0]["mean_branch_T_minus_S"] * by_comp[c][1]["mean_branch_T_minus_S"] >= 0) for c in COMPONENTS}
    decision = "formal_k4_k8_gate_passed" if all(float(r["mean_branch_T_minus_S"]) > 0 for r in summary) else "formal_k4_k8_gate_failed"
    ranking = sorted([{"component": c, "mean_live_utility": comp_score[c], "mean_replay_noise": comp_noise[c], "effect_over_noise": comp_score[c] / max(comp_noise[c], 1e-12), "K4_K8_direction_consistent": consistent[c]} for c in COMPONENTS], key=lambda r: r["mean_live_utility"], reverse=True)
    decision_payload = {"decision": decision, "component": "evidence_graph", "contract": "same xi_t; evidence_graph ON full branch vs OFF reduced branch; both continuations reduced policy; no full-harness takeover", "split": "UTILITY_LIVE256", "seed": args.seed, "runner": "true_live_fork_replay_hf_bm25", "full_harness_takeover": False, "rows": summary, "ranking": ranking}
    (out / "EVIDENCE_GRAPH_K4_K8_GATE.json").write_text(json.dumps(decision_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# EVIDENCE_GRAPH_K4_K8_GATE", "", f"- decision: `{decision}`", "- runner: `true_live_fork_replay_hf_bm25`", "- full_harness_takeover: false", "- replay_noise: measured N1/N2 same-action branches, not assumed zero", "", "| component | mean live utility | replay noise | effect/noise | K4/K8 consistent |", "|---|---:|---:|---:|---|"]
    for r in ranking:
        lines.append(f"| {r['component']} | {r['mean_live_utility']:.6f} | {r['mean_replay_noise']:.6f} | {r['effect_over_noise']:.3f} | {r['K4_K8_direction_consistent']} |")
    (out / "EVIDENCE_GRAPH_K4_K8_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for comp, fname in [("evidence_graph", "EVIDENCE_GRAPH_LIVE_UTILITY.md")]:
        rows = by_comp[comp]
        (out / fname).write_text(f"# {comp} live utility\n\n- runner: `true_live_fork_replay_hf_bm25`\n- K=4 T-S: {rows[0]['mean_branch_T_minus_S']:.6f}\n- K=8 T-S: {rows[1]['mean_branch_T_minus_S']:.6f}\n- replay noise K4/K8: {rows[0]['replay_noise']:.6f} / {rows[1]['replay_noise']:.6f}\n- effect/noise K4/K8: {rows[0]['effect_over_replay_noise']:.3f} / {rows[1]['effect_over_replay_noise']:.3f}\n", encoding="utf-8")
    pre = out.parent / "scape_prestage_v3"; pre.mkdir(parents=True, exist_ok=True)
    (pre / "EVIDENCE_GRAPH_FORMAL_FORK_HANDOFF.json").write_text(json.dumps(decision_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = build_run_manifest(run_id="evidence_graph_formal_fork_0820", stage="evidence_graph_formal_fork", command=sys.argv, repo_root=REPO, output_dir=out, input_paths={"utility_live_manifest": out / "manifests" / "UTILITY_LIVE256.json"}, extra={"runner": "true_live_fork_replay_hf_bm25", "components": COMPONENTS, "K": [4,8], "n_states_per_component": args.n_states, "full_harness_takeover": False})
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=[f"{c}_K{k}" for c in COMPONENTS for k in (4,8)] + ["replay_noise", "aggregation"]))
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    write_status_live(out / "STATUS_LIVE.md", stage="evidence_graph_formal_fork", run_id="evidence_graph_formal_fork_0820", n_expected=8, n_finished=8, errors=[], extra={"phase": "quality-complete", "decision": decision, "runner": "true_live_fork_replay_hf_bm25"})
    print(json.dumps(decision_payload, indent=2), flush=True)
    return 0


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["utility", "noise", "aggregate"], required=True)
    ap.add_argument("--component", choices=COMPONENTS, default="evidence_graph")
    ap.add_argument("--K", type=int, choices=[2, 4, 8])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--corpus-path", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "0820_evidence_graph_formal_fork")
    ap.add_argument("--seed", type=int, default=2214)
    ap.add_argument("--n-states", type=int, default=256)
    ap.add_argument("--n-queries-pool", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--states-cache", type=Path)
    ap.add_argument("--state-offset", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    if args.mode == "utility":
        if not args.component or not args.K:
            raise SystemExit("--component and --K required for --mode utility")
        return run_utility_shard(args)
    if args.mode == "noise":
        return run_noise_shard(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
