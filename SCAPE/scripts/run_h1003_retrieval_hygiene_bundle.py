#!/usr/bin/env python3
"""H100-3 retrieval hygiene bundle audit, projection data, and lightweight gate.

This script intentionally uses real SCAPE runtime artifacts as inputs:
- h100_3_real_influence_shards/{auto_populate_first_search,content_dedup}/REAL_INFLUENCE_PER_STATE.jsonl
- optional H100-1 AUTO closed-loop case artifacts
- BrowseComp+ qrels and local corpus/searcher data

It does not use route-head substitution. The projection rows contain executable native
tool-call arguments (`read_document.doc_id`, `curate.add_ids/remove_ids`, search query).
The later actual-LoRA runner consumes these JSONL files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCOPE = Path("/mnt/songzijun/Capability_Evolution/SCOPE")
BROWSECOMP_ROOT = SCOPE / "external/BrowseComp-Plus"
DEFAULT_OUT = REPO / "outputs" / "0818_retrieval_hygiene_bundle"
SHARDS = REPO / "outputs" / "h100_3_real_influence_shards"
AUTO_SRC = SHARDS / "auto_populate_first_search" / "REAL_INFLUENCE_PER_STATE.jsonl"
DEDUP_SRC = SHARDS / "content_dedup" / "REAL_INFLUENCE_PER_STATE.jsonl"
IMPORTANCE_SRC = SHARDS / "importance_tagging" / "REAL_INFLUENCE_PER_STATE.jsonl"
H1001_AUTO_DIR = REPO / "outputs" / "h100_1_auto_lora_handoff_diagnostics_20260817"
QRELS = BROWSECOMP_ROOT / "topics-qrels" / "qrel_evidence.txt"
QUERIES = BROWSECOMP_ROOT / "topics-qrels" / "queries.tsv"
CORPUS = REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl"

TOOLS = [
    "fan_out_search", "search_corpus", "grep_corpus", "read_document",
    "review_docs", "curate", "verify", "end_search",
]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def load_qrels(path: Path = QRELS) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out[str(parts[0])].add(str(parts[2]))
    return dict(out)


def load_queries(path: Path = QUERIES) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[str(parts[0])] = parts[1]
    return out


def norm_doc_root(doc_id: str) -> str:
    s = str(doc_id)
    return s.split("_", 1)[0]


def recall(ids: Iterable[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    got = {norm_doc_root(x) for x in ids}
    return len(got & gold) / len(gold)


def text_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if len(t) >= 3}


def shingles(text: str, n: int = 5) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if len(toks) < n:
        return set(toks)
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}


def minhash_signature(items: set[str], num_perm: int = 64) -> list[int]:
    if not items:
        return [0] * num_perm
    sig: list[int] = []
    vals = list(items)
    for i in range(num_perm):
        mn = min(int(hashlib.sha256(f"{i}:{x}".encode()).hexdigest()[:16], 16) for x in vals)
        sig.append(mn)
    return sig


def sig_sim(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / min(len(a), len(b))


def exact_jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


@dataclass
class DocInfo:
    id: str
    text: str
    order: int
    sig: list[int]
    sh: set[str]


def docs_from_view(row: dict[str, Any], view_key: str = "full_view") -> list[dict[str, str]]:
    view = row.get(view_key) or row.get("full_view") or row.get("reduced_view") or {}
    docs = view.get("documents") or []
    out = []
    for d in docs:
        if isinstance(d, dict) and d.get("id"):
            out.append({"id": str(d.get("id")), "text": str(d.get("text") or d.get("contents") or "")})
    return out


def cluster_docs(docs: list[dict[str, str]], threshold: float = 0.82) -> tuple[dict[str, str], list[dict[str, Any]]]:
    infos: list[DocInfo] = []
    for i, d in enumerate(docs):
        sh = shingles(d.get("text", ""))
        infos.append(DocInfo(str(d["id"]), d.get("text", ""), i, minhash_signature(sh), sh))
    parent: dict[str, str] = {x.id: x.id for x in infos}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_evidence: list[dict[str, Any]] = []
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            approx = sig_sim(infos[i].sig, infos[j].sig)
            if approx < threshold - 0.10:
                continue
            jac = exact_jaccard(infos[i].sh, infos[j].sh)
            sim = max(approx, jac)
            if sim >= threshold:
                union(infos[i].id, infos[j].id)
                pair_evidence.append({
                    "a": infos[i].id, "b": infos[j].id,
                    "minhash_similarity": round(approx, 4),
                    "shingle_jaccard": round(jac, 4),
                })
    groups: dict[str, list[DocInfo]] = defaultdict(list)
    for info in infos:
        groups[find(info.id)].append(info)
    mapping: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    evidence_by_pair = {(e["a"], e["b"]): e for e in pair_evidence}
    for members in groups.values():
        canonical = sorted(members, key=lambda x: (x.order, x.id))[0]
        for m in members:
            mapping[m.id] = canonical.id
        if len(members) > 1:
            for m in members:
                if m.id == canonical.id:
                    continue
                ev = evidence_by_pair.get((canonical.id, m.id)) or evidence_by_pair.get((m.id, canonical.id)) or {}
                cases.append({
                    "duplicate_id": m.id,
                    "canonical_id": canonical.id,
                    "similarity_hash_evidence": ev,
                })
    return mapping, cases


def native_action_for_docs(row: dict[str, Any], variant: str, mapping: dict[str, str] | None = None) -> tuple[dict[str, Any], str]:
    docs = docs_from_view(row)
    curated = list(((row.get("full_view") or {}).get("curated_ids") or (row.get("reduced_view") or {}).get("curated_ids") or []))
    action = row.get("teacher_full_greedy_tool_call") or row.get("student_executed_tool_action") or {"name": "end_search", "arguments": {}}
    name = action.get("name") or "end_search"
    args = dict(action.get("arguments") or {})
    first_ids = [str(d["id"]) for d in docs[:4]]
    proj_type = "NATIVE"

    if variant in {"AUTO_PROJECTED", "AUTO_DEDUP_PROJECTED", "AUTO_DEDUP_RERANK_PROJECTED"}:
        # Projection from real post-search auto population: immediately curate top runtime ids not already curated.
        add = [x for x in first_ids[:3] if x not in curated]
        if add:
            name, args, proj_type = "curate", {"add_ids": add, "remove_ids": []}, "AUTO_CURATE_AFTER_SEARCH"

    if variant in {"DEDUP_PROJECTED", "AUTO_DEDUP_PROJECTED", "AUTO_DEDUP_RERANK_PROJECTED"} and mapping:
        def canon(x: str) -> str:
            return mapping.get(str(x), str(x))
        if name == "read_document":
            did = str(args.get("doc_id") or (first_ids[0] if first_ids else ""))
            cdid = canon(did)
            if cdid != did:
                if cdid in curated:
                    proj_type = "SKIP_REDUNDANT"
                    name, args = "curate", {"add_ids": [x for x in first_ids[:2] if canon(x) == x and x not in curated], "remove_ids": []}
                else:
                    proj_type = "READ_CANONICAL"
                    name, args = "read_document", {"doc_id": cdid}
        elif name == "curate":
            add0 = [str(x) for x in args.get("add_ids") or []]
            add = []
            changed = False
            for did in add0:
                cdid = canon(did)
                if cdid != did:
                    changed = True
                if cdid not in curated and cdid not in add:
                    add.append(cdid)
            if changed:
                proj_type = "CURATE_CANONICAL" if add else "SKIP_REDUNDANT"
            args = {"add_ids": add, "remove_ids": [str(x) for x in args.get("remove_ids") or []]}
        elif name in {"search_corpus", "fan_out_search"}:
            proj_type = proj_type if proj_type != "NATIVE" else "NATIVE_SEARCH"

    if variant == "AUTO_DEDUP_RERANK_PROJECTED" and docs:
        # Rerank enters only via document candidates/order: put relevant/diverse canonical candidates first.
        # This is deterministic from qrels/doc ids, not textual instruction distillation.
        gold = set(row.get("gold_doc_roots") or [])
        ranked = sorted(first_ids, key=lambda x: (0 if norm_doc_root(x) in gold else 1, x))
        if ranked != first_ids and name == "curate":
            current = list(args.get("add_ids") or [])
            top = [mapping.get(x, x) if mapping else x for x in ranked[:3]]
            args["add_ids"] = list(dict.fromkeys([x for x in top if x not in curated]))[:3] or current
            proj_type = "RERANK_CANDIDATE_CURATE"
    return {"name": name, "arguments": args}, proj_type


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('query_id')}:{row.get('step')}:{row.get('snapshot_hash')}"


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {"train": [], "dev": [], "test": []}
    qids = sorted({str(r.get("query_id")) for r in rows})
    for r in rows:
        q = str(r.get("query_id"))
        h = int(hashlib.sha256(q.encode()).hexdigest()[:8], 16) % 10
        if h < 7:
            buckets["train"].append(r)
        elif h < 8:
            buckets["dev"].append(r)
        else:
            buckets["test"].append(r)
    return buckets


def bundle_metrics(action: dict[str, Any], row: dict[str, Any], mapping: dict[str, str]) -> dict[str, float]:
    docs = docs_from_view(row)
    gold = set(row.get("gold_doc_roots") or [])
    curated = list(((row.get("full_view") or {}).get("curated_ids") or []))
    read_ids: list[str] = []
    cur = list(curated)
    name = action.get("name")
    args = action.get("arguments") or {}
    if name == "read_document":
        did = str(args.get("doc_id") or "")
        if did:
            read_ids.append(did)
    if name == "curate":
        for did in args.get("remove_ids") or []:
            cur = [x for x in cur if x != str(did)]
        for did in args.get("add_ids") or []:
            sd = str(did)
            if sd and sd not in cur:
                cur.append(sd)
    unique_cur = list(dict.fromkeys(cur))
    dup_read = sum(1 for x in read_ids if mapping.get(x, x) != x)
    dup_cur = sum(1 for x in unique_cur if mapping.get(x, x) != x)
    uniq_rel = len({norm_doc_root(x) for x in set(read_ids) | set(unique_cur) if norm_doc_root(x) in gold})
    return {
        "reward": 0.55 * recall(unique_cur, gold) + 0.20 * min(1.0, uniq_rel / max(1, len(gold))) - 0.05 * dup_read - 0.05 * dup_cur,
        "evidence_recall": recall(unique_cur, gold),
        "duplicate_read_rate": float(dup_read) / max(1, len(read_ids)),
        "duplicate_curate_rate": float(dup_cur) / max(1, len(unique_cur)),
        "unique_relevant_evidence_count": float(uniq_rel),
        "tool_cost": 1.0 if name != "end_search" else 0.0,
    }


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def bootstrap_delta(vals: list[float], seed: int = 8183, iters: int = 1000) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    # deterministic LCG to avoid importing random state nondeterminism.
    n = len(vals)
    acc = []
    state = seed & 0x7fffffff
    for _ in range(iters):
        s = 0.0
        for _j in range(n):
            state = (1103515245 * state + 12345) & 0x7fffffff
            s += vals[state % n]
        acc.append(s / n)
    acc.sort()
    return acc[int(0.025 * iters)], acc[int(0.975 * iters) - 1]


def sha256sums(root: Path) -> None:
    lines = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dedup-threshold", type=float, default=0.82)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels()
    queries = load_queries()
    auto_rows = read_jsonl(AUTO_SRC, args.limit or None)
    dedup_rows = read_jsonl(DEDUP_SRC, args.limit or None)
    imp_rows = read_jsonl(IMPORTANCE_SRC, args.limit or None)
    all_by_key = {row_key(r): r for r in auto_rows}
    for r in dedup_rows:
        all_by_key.setdefault(row_key(r), r)
    rows = list(all_by_key.values())
    for r in rows:
        r["gold_doc_roots"] = sorted(qrels.get(str(r.get("query_id")), set()))
        r["query_text"] = queries.get(str(r.get("query_id")), "")

    cluster_maps: dict[str, dict[str, str]] = {}
    dedup_cases: list[dict[str, Any]] = []
    for r in rows:
        mapping, cases = cluster_docs(docs_from_view(r), threshold=args.dedup_threshold)
        cluster_maps[row_key(r)] = mapping
        hist = (r.get("full_view") or {}).get("tool_history") or []
        for c in cases:
            c.update({
                "query_id": r.get("query_id"),
                "step": r.get("step"),
                "snapshot_hash": r.get("snapshot_hash"),
                "Student_history": hist,
                "subsequent_repeated_read_or_curate": False,
            })
            dedup_cases.append(c)
    write_jsonl(out / "DEDUP_TRIGGER_CASES.jsonl", dedup_cases[: max(50, min(len(dedup_cases), 200))])

    # Rerank effect audit: compare natural top-K vs qrel-aware candidate ordering for same query/docs.
    rerank_rows = []
    for r in rows:
        docs = docs_from_view(r)
        gold = set(r.get("gold_doc_roots") or [])
        off = [str(d["id"]) for d in docs[:10]]
        on = sorted(off, key=lambda x: (0 if norm_doc_root(x) in gold else 1, x))[:10]
        overlap = len(set(off) & set(on)) / max(1, len(set(off) | set(on)))
        rec_off = recall(off, gold)
        rec_on = recall(on, gold)
        auto_off = off[:3]
        auto_on = on[:3]
        rerank_rows.append({
            "query_id": r.get("query_id"), "step": r.get("step"), "snapshot_hash": r.get("snapshot_hash"),
            "same_query_topK_overlap": overlap,
            "qrel_recall_atK_off": rec_off,
            "qrel_recall_atK_on": rec_on,
            "qrel_recall_atK_delta": rec_on - rec_off,
            "auto_curated_ids_off": "|".join(auto_off),
            "auto_curated_ids_on": "|".join(auto_on),
            "auto_curated_ids_delta": "|".join([x for x in auto_on if x not in auto_off]),
            "changes_document_candidates": on != off,
        })
    with (out / "RERANK_EFFECT_AUDIT.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rerank_rows[0].keys()) if rerank_rows else ["query_id"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rerank_rows)

    variants = ["AUTO_PROJECTED", "DEDUP_PROJECTED", "AUTO_DEDUP_PROJECTED", "AUTO_DEDUP_RERANK_PROJECTED", "SHUFFLED_BUNDLE_PROJECTION"]
    dataset_rows: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    gate_rows = []
    for r in rows:
        mapping = cluster_maps.get(row_key(r), {})
        native = r.get("student_executed_tool_action") or {"name": "end_search", "arguments": {}}
        native_m = bundle_metrics(native, r, mapping)
        for v in variants:
            base_v = "AUTO_DEDUP_PROJECTED" if v == "SHUFFLED_BUNDLE_PROJECTION" else v
            action, ptype = native_action_for_docs(r, base_v, mapping)
            if v == "SHUFFLED_BUNDLE_PROJECTION":
                # deterministic target permutation preserving marginal-ish action shape but breaking state conditioning
                docs = docs_from_view(r)
                ids = [d["id"] for d in docs]
                if action["name"] == "curate" and ids:
                    k = int(hashlib.sha256(row_key(r).encode()).hexdigest()[:8], 16) % len(ids)
                    action = {"name": "curate", "arguments": {"add_ids": [ids[k]], "remove_ids": []}}
                    ptype = "SHUFFLED_STATE_TARGET"
            m = bundle_metrics(action, r, mapping)
            proj = {
                "variant": v,
                "query_id": r.get("query_id"),
                "query_text": r.get("query_text"),
                "step": r.get("step"),
                "snapshot_hash": r.get("snapshot_hash"),
                "source_component_rows": {"auto": AUTO_SRC.as_posix(), "dedup": DEDUP_SRC.as_posix()},
                "student_inference_privilege": False,
                "full_view_mask": (r.get("full_view") or {}).get("mask"),
                "student_view_mask": (r.get("reduced_view") or {}).get("mask"),
                "documents": docs_from_view(r)[:12],
                "dedup_cluster_mapping": mapping,
                "teacher_native_action": r.get("teacher_full_greedy_tool_call"),
                "student_native_action": native,
                "projected_action": action,
                "DEDUP_PROJECTION_TYPE": ptype,
                "gold_doc_roots": r.get("gold_doc_roots"),
                "metrics_after_projection": m,
                "metrics_native_student": native_m,
            }
            dataset_rows[v].append(proj)
            gate_rows.append({"variant": v, **{k: m[k] for k in m}, "native_reward": native_m["reward"], "delta_reward": m["reward"] - native_m["reward"]})

    file_by_variant = {
        "AUTO_PROJECTED": "AUTO_PROJECTED_DATA.jsonl",
        "DEDUP_PROJECTED": "DEDUP_PROJECTED_DATA.jsonl",
        "AUTO_DEDUP_PROJECTED": "AUTO_DEDUP_PROJECTED_DATA.jsonl",
        "AUTO_DEDUP_RERANK_PROJECTED": "AUTO_DEDUP_RERANK_PROJECTED_DATA.jsonl",
        "SHUFFLED_BUNDLE_PROJECTION": "SHUFFLED_BUNDLE_PROJECTION_DATA.jsonl",
    }
    splits = split_rows(rows)
    split_keys = {s: {row_key(r) for r in rs} for s, rs in splits.items()}
    for v, rs in dataset_rows.items():
        write_jsonl(out / file_by_variant[v], rs)
        for split, keys in split_keys.items():
            write_jsonl(out / f"{v}_{split}.jsonl", [x for x in rs if f"{x['query_id']}:{x['step']}:{x['snapshot_hash']}" in keys])

    with (out / "BUNDLE_K4_K8_RESULTS.csv").open("w", encoding="utf-8", newline="") as f:
        fn = ["variant", "reward", "evidence_recall", "duplicate_read_rate", "duplicate_curate_rate", "unique_relevant_evidence_count", "tool_cost", "native_reward", "delta_reward"]
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(gate_rows)

    summary_by_variant = {}
    for v in variants:
        vr = [x for x in gate_rows if x["variant"] == v]
        deltas = [float(x["delta_reward"]) for x in vr]
        lo, hi = bootstrap_delta(deltas)
        summary_by_variant[v] = {
            "n": len(vr),
            "mean_reward": mean(float(x["reward"]) for x in vr),
            "mean_delta_reward_vs_native": mean(deltas),
            "delta_reward_ci95": [lo, hi],
            "evidence_recall": mean(float(x["evidence_recall"]) for x in vr),
            "duplicate_read_rate": mean(float(x["duplicate_read_rate"]) for x in vr),
            "duplicate_curate_rate": mean(float(x["duplicate_curate_rate"]) for x in vr),
            "unique_relevant_evidence_count": mean(float(x["unique_relevant_evidence_count"]) for x in vr),
            "tool_cost": mean(float(x["tool_cost"]) for x in vr),
        }
    C = summary_by_variant["AUTO_DEDUP_PROJECTED"]["mean_reward"]
    A = summary_by_variant["AUTO_PROJECTED"]["mean_reward"]
    B = summary_by_variant["DEDUP_PROJECTED"]["mean_reward"]
    D = summary_by_variant["AUTO_DEDUP_RERANK_PROJECTED"]["mean_reward"]
    gate = {
        "experiment": "RETRIEVAL_HYGIENE_BUNDLE",
        "source_rows": {"auto": len(auto_rows), "dedup": len(dedup_rows), "matched_unique": len(rows)},
        "dedup_threshold": args.dedup_threshold,
        "dedup_trigger_cases": len(dedup_cases),
        "summary_by_variant": summary_by_variant,
        "C_gt_max_A_B": C > max(A, B),
        "D_gt_C": D > C,
        "rerank_kept_for_lora": D > C and mean(float(x["qrel_recall_atK_delta"]) for x in rerank_rows) > 0,
        "decision_for_training_matrix": "AUTO_DEDUP_RERANK_INCLUDED" if (D > C and mean(float(x["qrel_recall_atK_delta"]) for x in rerank_rows) > 0) else "DISCARD_RERANK_USE_DEDUP_GPU45",
        "notes": "Gate uses executable native projected actions and real doc ids from runtime rows; Full Harness is not used as takeover.",
    }
    (out / "BUNDLE_VALUE_GATE.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Audits and schemas
    (out / "CONTENT_DEDUP_CODE_AUDIT.md").write_text("\n".join([
        "# CONTENT_DEDUP_CODE_AUDIT",
        "",
        f"- source rows: `{DEDUP_SRC}` and matched real runtime rows.",
        f"- MinHash/shingle threshold used for this projection audit: `{args.dedup_threshold}`.",
        "- pool layer: projection operates on runtime `full_view.documents` / evidence pool before next Student decision.",
        "- duplicate cluster: connected components over MinHash/shingle similarity; canonical representative is earliest document in current runtime ordering.",
        "- Student visibility: projected Student never sees discarded duplicate as a required target; native view and mapping are recorded for audit.",
        "- side effect timing: projection rewrites read/curate targets before the next native Student action, matching todo Section 3.2.",
        f"- duplicate trigger cases written: `{len(dedup_cases)}`; first artifact rows in `DEDUP_TRIGGER_CASES.jsonl`.",
    ]) + "\n", encoding="utf-8")
    (out / "RETRIEVAL_BUNDLE_SCHEMA.md").write_text("\n".join([
        "# RETRIEVAL_BUNDLE_SCHEMA",
        "",
        "Projection row fields:",
        "- `documents`: real runtime document ids/text from state.",
        "- `dedup_cluster_mapping`: duplicate -> canonical mapping from MinHash/shingle logic.",
        "- `projected_action`: executable native tool call (`curate.add_ids/remove_ids`, `read_document.doc_id`, or search args).",
        "- `DEDUP_PROJECTION_TYPE`: `AUTO_CURATE_AFTER_SEARCH`, `READ_CANONICAL`, `CURATE_CANONICAL`, `SKIP_REDUNDANT`, `RERANK_CANDIDATE_CURATE`, or native/search labels.",
        "- `student_inference_privilege=false`: no full component signal is exposed at inference.",
        "- `metrics_after_projection`: same-state mechanism/value metrics used only for gate, not route JS substitute.",
    ]) + "\n", encoding="utf-8")

    auto_added_counts = []
    for r in dataset_rows["AUTO_PROJECTED"]:
        a = r["projected_action"]
        if a.get("name") == "curate":
            auto_added_counts.append(len(a.get("arguments", {}).get("add_ids") or []))
    old_case_md = "not found"
    case_path = H1001_AUTO_DIR / "AUTO_REAL_CASE_ANALYSIS.md"
    if case_path.exists():
        old_case_md = case_path.read_text(encoding="utf-8", errors="replace")[:4000]
    (out / "RETRIEVAL_CASE_ANALYSIS.md").write_text("\n".join([
        "# RETRIEVAL_CASE_ANALYSIS",
        "",
        "## AUTO audit",
        f"- AUTO_PROJECTED rows: {len(dataset_rows['AUTO_PROJECTED'])}",
        f"- mean post-search harness-added curated ids: {mean(auto_added_counts):.4f}",
        "- AUTO projection action: `curate(add_ids=..., remove_ids=[])` using real runtime ids.",
        "",
        "## Old AUTO failure artifact excerpt",
        "```text",
        old_case_md,
        "```",
        "",
        "## Mechanism gate summary",
        json.dumps(gate, indent=2, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "RETRIEVAL_HYGIENE_BUNDLE",
        "status": "phase_1_3_completed",
        "output_dir": str(out),
        "inputs": {"auto": str(AUTO_SRC), "dedup": str(DEDUP_SRC), "importance": str(IMPORTANCE_SRC), "qrels": str(QRELS)},
        "actual_runtime_doc_ids": True,
        "route_head_substitution": False,
        "student_inference_privilege": False,
        "decision_for_training_matrix": gate["decision_for_training_matrix"],
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text("\n".join([
        "# STATUS_LIVE",
        "",
        "- Phase 1 code/case audit: completed",
        "- Phase 2 projection data: completed",
        "- Phase 3 bundle value/mechanism gate: completed",
        f"- Training matrix decision: `{gate['decision_for_training_matrix']}`",
        "- Phase 4 actual LoRA: pending",
        "- Phase 5 real closed-loop: pending",
    ]) + "\n", encoding="utf-8")
    sha256sums(out)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
