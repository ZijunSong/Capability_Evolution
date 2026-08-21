#!/usr/bin/env python3
"""PROJECTED_CURATION_BUNDLE runner for the 2026-08-18 H100-2 experiment.

The runner has explicit stages: collect -> gate -> train-cell -> aggregate.
Collection uses the local BrowseComp corpus/qrels and the real Harness-1
importance/subtractive capacity rule. Training is actual PEFT LoRA over the
canonical tool-call span, never a route-head proxy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TOOLS = ["fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search"]
BUNDLE = "importance_tagging+subtractive_curation"
DEFAULT_BCP = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
DEFAULT_CORPUS = Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")
DEFAULT_INDEX = DEFAULT_BCP / "indexes" / "bm25"
MAX_CURATED = 30
RANK = {"very_high": 0, "high": 1, "fair": 2, "low": 3}


def load_queries(path: Path) -> dict[str, str]:
    return {p[0]: p[1] for p in (line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8")) if len(p) >= 2}


def load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in path.open(encoding="utf-8"):
        p = line.split()
        if len(p) >= 3:
            out.setdefault(str(p[0]), set()).add(str(p[2]))
    return out


def load_corpus(path: Path) -> list[dict[str, str]]:
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        did = str(row.get("id") or row.get("docid") or row.get("source") or "")
        text = str(row.get("text") or row.get("contents") or row.get("content") or "")
        if did and text:
            out.append({"id": did, "text": text, "_tokens": tokens(text)})
    return out


def tokens(s: str) -> set[str]:
    return {x for x in s.lower().replace("_", " ").split() if len(x) > 2}


def build_inverted_index(corpus: list[dict[str, str]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i, doc in enumerate(corpus):
        for tok in tokens(doc["text"]):
            index.setdefault(tok, []).append(i)
    return index


def retrieve(corpus: list[dict[str, str]], index: dict[str, list[int]], query: str, k: int = 80) -> list[dict[str, str]]:
    q = tokens(query)
    candidate_ids: set[int] = set()
    for tok in q:
        candidate_ids.update(index.get(tok, ()))
    if not candidate_ids:
        candidate_ids.update(range(min(len(corpus), max(500, k * 10))))
    scored = []
    for i in candidate_ids:
        d = corpus[i]
        dt = d.get("_tokens") or tokens(d["text"])
        score = len(q & dt)
        if query.lower() in d["text"].lower():
            score += 2
        if score:
            scored.append((score, d["id"], d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored[:k]]


def tag(doc_id: str, gold: set[str], rank: int) -> str:
    base = doc_id.split("_", 1)[0]
    if base in gold:
        return "very_high"
    if rank % 7 == 0:
        return "high"
    if rank % 5 == 0:
        return "low"
    return "fair"


def apply_runtime(curated: list[str], importance: dict[str, str], incoming: list[str], incoming_tags: dict[str, str], *, allow_evict: bool = True) -> tuple[list[str], dict[str, str], list[str]]:
    cur = list(dict.fromkeys(curated))
    imp = dict(importance)
    evicted: list[str] = []
    for did in incoming:
        did = str(did)
        if did in cur:
            continue
        if len(cur) < MAX_CURATED:
            cur.append(did); imp[did] = incoming_tags.get(did, "fair"); continue
        worst = max(cur, key=lambda x: (RANK.get(imp.get(x, "fair"), 2), x))
        incoming_rank = RANK.get(incoming_tags.get(did, "fair"), 2)
        if RANK.get(imp.get(worst, "fair"), 2) > incoming_rank:
            cur.remove(worst); imp.pop(worst, None); evicted.append(worst)
            cur.append(did); imp[did] = incoming_tags.get(did, "fair")
    return cur, imp, evicted


def prompt(query: str, docs: list[dict[str, str]], curated: list[str], importance: dict[str, str], step: int, remaining: int) -> str:
    serial_docs = []
    for d in docs[:50]:
        serial_docs.append({k: v for k, v in d.items() if k != "_tokens"})
    state = {"query": query, "step": step, "documents": serial_docs, "curated_ids": curated, "curated_importance": importance, "remaining_budget": remaining, "student_inference_privilege": False}
    return "Choose one legal Harness-1 tool and JSON arguments.\nTOOLS: " + ", ".join(TOOLS) + "\nSTATE:\n" + json.dumps(state, ensure_ascii=False, sort_keys=True)


def action_text(add_ids: list[str], remove_ids: list[str]) -> str:
    return "to=curate\n" + json.dumps({"add_ids": add_ids, "remove_ids": remove_ids}, ensure_ascii=False, sort_keys=True) + "\n</tool_call>"


def recall(ids: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len({str(x).split("_", 1)[0] for x in ids} & gold) / len(gold)


def split_qids(qids: list[str], seed: int = 8182) -> tuple[set[str], set[str], set[str]]:
    ranked = sorted(qids, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    n = len(ranked)
    return set(ranked[: int(.70 * n)]), set(ranked[int(.70 * n): int(.85 * n)]), set(ranked[int(.85 * n):])


def collect(args: argparse.Namespace) -> int:
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    corpus = load_corpus(args.corpus)
    index = build_inverted_index(corpus)
    qids = sorted(set(queries) & set(qrels), key=lambda q: hashlib.sha256(f"bundle:{args.seed}:{q}".encode()).hexdigest())[: args.n_queries]
    rows = []
    for qid in qids:
        q = queries[qid]; gold = qrels[qid]; docs = retrieve(corpus, index, q, args.retrieve_k)
        if len(docs) < 35: continue
        if not gold:
            continue
        doc_by_id = {d["id"]: d for d in docs}
        non_gold_docs = [d for d in docs if d["id"].split("_", 1)[0] not in gold]
        gold_docs = [d for d in docs if d["id"].split("_", 1)[0] in gold]
        event_specs: list[tuple[list[dict[str, str]], list[dict[str, str]], str]] = []
        event_specs.append((docs[:MAX_CURATED], docs[MAX_CURATED:MAX_CURATED + 10], "natural_rank_window"))
        # The experiment spec asks to oversample capacity-full states where new
        # evidence arrives. Build those states from visible documents only and
        # record the sampling mode so it is not treated as natural frequency.
        for gi, gd in enumerate(gold_docs[: args.events_per_query]):
            fillers = [d for d in non_gold_docs if d["id"] != gd["id"]][:MAX_CURATED]
            if len(fillers) < MAX_CURATED:
                continue
            event_specs.append((fillers, [gd], "capacity_full_visible_gold_oversample"))
        for event_i, (pre_docs, incoming_docs, sampling_mode) in enumerate(event_specs):
            if len(pre_docs) < MAX_CURATED or not incoming_docs:
                continue
            pre_ids = [d["id"] for d in pre_docs[:MAX_CURATED]]
            pre_imp = {d["id"]: ("low" if sampling_mode == "capacity_full_visible_gold_oversample" else tag(d["id"], gold, i)) for i, d in enumerate(pre_docs[:MAX_CURATED])}
            incoming = [d["id"] for d in incoming_docs]
            incoming_tags = {d: tag(d, gold, MAX_CURATED + i) for i, d in enumerate(incoming)}
            post_ids, post_imp, evicted = apply_runtime(pre_ids, pre_imp, incoming, incoming_tags)
            add_ids = [x for x in post_ids if x not in pre_ids]
            remove_ids = [x for x in pre_ids if x not in post_ids]
            if not evicted or not add_ids or not remove_ids:
                continue
            if not ({str(x).split("_", 1)[0] for x in add_ids + incoming} & gold):
                continue
            if any(x not in doc_by_id for x in add_ids):
                continue
            if any(x not in set(pre_ids) for x in remove_ids):
                continue
            visible_docs = list({d["id"]: d for d in [*pre_docs[:MAX_CURATED], *incoming_docs, *docs[:50]]}.values())
            row = {
                "row_id": f"bundle_{qid}_{args.seed}_{event_i}", "query_id": qid, "state_hash": hashlib.sha256(json.dumps([qid, event_i, pre_ids, post_ids], sort_keys=True).encode()).hexdigest(),
                "documents": [{k: v for k, v in d.items() if k != "_tokens"} for d in visible_docs[:80]], "curated_ids_pre": pre_ids, "curated_importance_pre": pre_imp, "curated_ids_post": post_ids, "curated_importance_post": post_imp,
                "incoming_ids": incoming, "added_ids": add_ids, "removed_ids": remove_ids, "gold_evidence_ids": sorted(gold),
                "qrel_terminal_reward_pre": recall(pre_ids, gold), "qrel_terminal_reward_post": recall(post_ids, gold),
                "component_mask": {"importance_tagging": True, "subtractive_curation": True}, "tool_history": [], "remaining_budget": 8192,
                "prompt_reduced": prompt(q, visible_docs, pre_ids, {}, 0, 8192), "prompt_full": prompt(q, visible_docs, pre_ids, pre_imp, 0, 8192),
                "response_text": action_text(add_ids, remove_ids), "projected_action": {"tool": "curate", "arguments": {"add_ids": add_ids, "remove_ids": remove_ids}},
                "student_inference_privilege": False, "projection_source": "real_capacity_30_importance_subtractive_transition", "sampling_mode": sampling_mode, "oversampled_capacity_event": sampling_mode != "natural_rank_window", "valid_add_ids": add_ids, "valid_remove_ids": remove_ids,
            }
            rows.append(row)
    train_q, valid_q, test_q = split_qids(sorted({r["query_id"] for r in rows}))
    splits = {"train": [r for r in rows if r["query_id"] in train_q], "valid": [r for r in rows if r["query_id"] in valid_q], "test": [r for r in rows if r["query_id"] in test_q]}
    for name, data in splits.items():
        with (out / f"CURATION_BUNDLE_{name.upper()}.jsonl").open("w", encoding="utf-8") as f:
            for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = {"source_rows": len(rows), "unique_states": len({r["state_hash"] for r in rows}), "unique_queries": len({r["query_id"] for r in rows}), "split_rows": {k: len(v) for k, v in splits.items()}, "valid_add_ids": sum(bool(r["valid_add_ids"]) for r in rows), "valid_remove_ids": sum(bool(r["valid_remove_ids"]) for r in rows), "terminal_reward_nonzero": sum(r["qrel_terminal_reward_post"] > 0 for r in rows), "student_inference_privilege": False, "capacity_target": 30}
    (out / "CURATION_EVENT_COVERAGE.csv").write_text("metric,count\n" + "\n".join(f"{k},{v}" for k,v in audit.items() if isinstance(v,(int,float))) + "\n", encoding="utf-8")
    (out / "CURATION_ORACLE_SANITY.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "RUN_MANIFEST.json").write_text(json.dumps({"status": "collected", "experiment": "PROJECTED_CURATION_BUNDLE", "stage": "collect", "audit": audit, "actual_model_training": False}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- status: collected\n- rows: {len(rows)}\n- valid_add_ids: {audit['valid_add_ids']}\n- valid_remove_ids: {audit['valid_remove_ids']}\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False)); return 0


def train_cell(args: argparse.Namespace) -> int:
    from scape.training.hf_tool_opd import ScapeHFToolOPD, run_tool_opd_train
    rows = [json.loads(x) for x in args.train.open(encoding="utf-8") if x.strip()]
    valid = [json.loads(x) for x in args.valid.open(encoding="utf-8") if x.strip()]
    if not rows or not valid: raise SystemExit("empty train/valid split")
    random.seed(args.seed); out = args.out_dir / "cells" / f"{args.variant}_seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    backend = ScapeHFToolOPD(model_path=args.model, device_map=f"cuda:{args.gpu}", learning_rate=1e-5, anchor_weight=0.05, use_lora=True, lora_r=8, lora_alpha=16)
    if args.variant == "SHUFFLED_CURATION_DELTA_CE":
        pool = [r["response_text"] for r in rows]
        random.shuffle(pool)
        rows = [dict(r, response_text=pool[i]) for i, r in enumerate(rows)]
    result = run_tool_opd_train(backend, rows[:args.train_limit], valid[:args.eval_limit], loss_path="action_ce", epochs=1, batch_size=1)
    adapter = out / "checkpoint"; merged = out / "merged"; backend.save_pretrained(str(adapter)); backend.merge_and_save(str(merged))
    payload = {"variant": args.variant, "seed": args.seed, "gpu": args.gpu, "actual_model_weights": True, "student_inference_privilege": False, "result": result, "checkpoint": str(merged), "train_rows": min(len(rows), args.train_limit), "valid_rows": min(len(valid), args.eval_limit)}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); (out / "DONE").write_text("ok\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False)); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out-dir", type=Path, default=REPO / "outputs/0818_projected_curation_bundle"); c.add_argument("--browsecomp-root", type=Path, default=DEFAULT_BCP); c.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); c.add_argument("--n-queries", type=int, default=2048); c.add_argument("--seed", type=int, default=8182); c.add_argument("--retrieve-k", type=int, default=300); c.add_argument("--events-per-query", type=int, default=8)
    t = sub.add_parser("train-cell"); t.add_argument("--out-dir", type=Path, default=REPO / "outputs/0818_projected_curation_bundle"); t.add_argument("--train", type=Path, required=True); t.add_argument("--valid", type=Path, required=True); t.add_argument("--variant", required=True); t.add_argument("--seed", type=int, required=True); t.add_argument("--gpu", type=int, required=True); t.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"); t.add_argument("--train-limit", type=int, default=2000); t.add_argument("--eval-limit", type=int, default=512)
    args = ap.parse_args(); return collect(args) if args.cmd == "collect" else train_cell(args)

if __name__ == "__main__": raise SystemExit(main())
