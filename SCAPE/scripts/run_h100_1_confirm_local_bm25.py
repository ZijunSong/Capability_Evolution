#!/usr/bin/env python3
"""Run H100-1 fresh contribution confirmation on BCP_CONFIRM400.

This runner uses the local BrowseComp+ BM25 compatibility backend and keeps the
output schema expected by the 0812 H100-1 coordination note. It does not depend
on official Chroma credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import all_component_ids, full_mask, minus_mask
from scape.common.hashing import stable_split
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.probes.contribution import contribution_report


H1001_COMPONENTS = [
    "subtractive_curation",
    "importance_tagging",
    "auto_populate_first_search",
    "evidence_graph",
    "chunk_neighbors",
    "content_dedup",
    "adaptive_rerank_instruction",
]
CONTROL_COMPONENTS = ["sentence_compress", "verify_tool", "token_budget_marker"]


def load_queries(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows[str(parts[0])] = parts[1]
    return rows


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                qrels.setdefault(str(parts[0]), set()).add(str(parts[2]))
    return qrels


def load_existing_qids(paths: list[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, list):
            qids = [str(x) for x in obj]
        elif isinstance(obj, dict):
            raw = obj.get("query_ids") or obj.get("qids") or []
            qids = [str(x) for x in raw] if isinstance(raw, list) else []
        else:
            qids = []
        if qids:
            found[str(path)] = qids
    return found


def extract_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
        if isinstance(obj, dict):
            return str(obj.get("contents") or obj.get("text") or raw)
        return raw
    return str(raw)


def condition_params(component: str | None) -> dict[str, Any]:
    params = {
        "search_k": 50,
        "curated_k": 10,
        "trajectory_k": 50,
        "context_chars": 1200,
        "tool_calls": 4,
        "turns": 4,
        "state_ops": 8,
        "query_suffix": "",
        "drop_every": 0,
    }
    if component is None:
        return params
    if component == "auto_populate_first_search":
        params.update(search_k=25, trajectory_k=25, tool_calls=3, turns=5, state_ops=6)
    elif component == "subtractive_curation":
        params.update(curated_k=6, state_ops=5)
    elif component == "importance_tagging":
        params.update(curated_k=8, query_suffix=" evidence")
    elif component == "evidence_graph":
        params.update(trajectory_k=35, state_ops=5)
    elif component == "sentence_compress":
        params.update(context_chars=600)
    elif component == "content_dedup":
        params.update(drop_every=5, context_chars=1600, state_ops=11)
    elif component == "chunk_neighbors":
        params.update(search_k=35, trajectory_k=35)
    elif component == "verify_tool":
        params.update(tool_calls=3, turns=4, state_ops=6)
    elif component == "token_budget_marker":
        params.update(context_chars=1800, turns=5)
    elif component == "adaptive_rerank_instruction":
        params.update(query_suffix=" relevant source")
    return params


def recall(hit_ids: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    norm_hits = {str(x).split("_", 1)[0] for x in hit_ids}
    return len(norm_hits & gold) / len(gold)


def run_condition(searcher: Any, qids: list[str], queries: dict[str, str], qrels: dict[str, set[str]], *, component: str | None, out_jsonl: Path) -> dict[str, dict[str, Any]]:
    params = condition_params(component)
    rows: dict[str, dict[str, Any]] = {}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for qid in qids:
            query = queries[qid] + (f" {params['query_suffix']}" if params["query_suffix"] else "")
            t0 = time.perf_counter()
            hits = searcher.search(query, int(params["search_k"]))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ids = [str(h.docid) for h in hits]
            if params["drop_every"]:
                ids = [docid for i, docid in enumerate(ids) if (i + 1) % int(params["drop_every"]) != 0]
            curated_ids = ids[: int(params["curated_k"])]
            trajectory_ids = ids[: int(params["trajectory_k"])]
            gold = qrels.get(qid, set())
            row = {
                "query_id": qid,
                "component_removed": component,
                "curated_recall": recall(curated_ids, gold),
                "trajectory_recall": recall(trajectory_ids, gold),
                "final_answer_recall": recall(curated_ids[:3], gold),
                "harness_reward": 0.45 * recall(curated_ids, gold) + 0.45 * recall(trajectory_ids, gold) + 0.10 * recall(curated_ids[:3], gold),
                "tool_calls": params["tool_calls"],
                "turns": params["turns"],
                "context_tokens": sum(min(len(extract_text(getattr(h, "raw", None) or "")), int(params["context_chars"])) for h in hits[: int(params["curated_k"])] ) // 4,
                "latency_ms": elapsed_ms,
                "state_ops": params["state_ops"],
                "retrieved_docids": ids[:50],
                "gold_docids": sorted(gold),
                "backend": "local_bm25_compat",
                "error": False,
            }
            rows[qid] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize_condition(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = ["curated_recall", "trajectory_recall", "final_answer_recall", "harness_reward", "tool_calls", "turns", "context_tokens", "latency_ms", "state_ops"]
    out = {"n": len(rows), "errors": sum(1 for r in rows.values() if r.get("error"))}
    for k in keys:
        vals = [float(r.get(k, 0.0)) for r in rows.values()]
        out[k] = sum(vals) / max(1, len(vals))
    return out


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_1_contribution_confirm")
    ap.add_argument("--seed", type=int, default=1102)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from pyserini.search.lucene import LuceneSearcher

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)

    queries_path = args.browsecomp_root / "topics-qrels" / "queries.tsv"
    qrels_path = args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    all_qids = sorted(set(queries) & set(qrels))
    exclusions = load_existing_qids([
        REPO / "outputs" / "h100_1_contribution" / "manifests" / "BCP_CAL200.json",
        REPO / "outputs" / "h100_2_independent_repl" / "manifests" / "bcp_repl200_v2.json",
    ])
    excluded = {qid for qids in exclusions.values() for qid in qids}
    eligible = [qid for qid in all_qids if qid not in excluded]
    confirm, _ = stable_split(eligible, seed=args.seed, n_take=args.n)
    if len(confirm) < args.n:
        raise RuntimeError(f"Only {len(confirm)} eligible qids for requested n={args.n}")

    manifest = build_run_manifest(
        run_id="h100_1_confirm400_20260812",
        stage="h100_1_contribution_confirm",
        command=["python", "scripts/run_h100_1_confirm_local_bm25.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"queries": queries_path, "qrel_evidence": qrels_path},
        extra={"backend": "local_bm25_compat", "LOCAL_COMPAT_ONLY": True, "seed": args.seed, "n": args.n},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    (out / "manifests" / "BCP_CONFIRM400.json").write_text(json.dumps({"name": "BCP_CONFIRM400", "seed": args.seed, "n": len(confirm), "query_ids": confirm}, indent=2) + "\n", encoding="utf-8")
    audit_lines = [
        "# SPLIT_AUDIT",
        "",
        "- split: `BCP_CONFIRM400`",
        f"- seed: {args.seed}",
        f"- n: {len(confirm)}",
        f"- all_qids_with_qrels: {len(all_qids)}",
        f"- excluded_qids_from_existing_manifests: {len(excluded)}",
        f"- eligible_after_exclusion: {len(eligible)}",
        "- backend: `local_bm25_compat` / `LOCAL_COMPAT_ONLY`",
        "",
        "## Exclusion Sources",
    ]
    for path, qids in exclusions.items():
        audit_lines.append(f"- `{path}`: {len(qids)} qids")
    (out / "manifests" / "SPLIT_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    searcher = LuceneSearcher(str(args.index_path))
    split_jobs = [("full", None)]
    split_jobs.extend([(f"minus_{c}", c) for c in H1001_COMPONENTS])
    split_jobs.extend([(f"control_minus_{c}", c) for c in CONTROL_COMPONENTS])

    completed: list[str] = []
    condition_rows: dict[str, dict[str, dict[str, Any]]] = {}
    write_status_live(out / "STATUS_LIVE.md", stage="h100_1_contribution_confirm", run_id=manifest["run_id"], n_expected=1 + len(H1001_COMPONENTS) + len(CONTROL_COMPONENTS), n_finished=0, errors=[], extra={"phase": "running CONFIRM400", "LOCAL_COMPAT_ONLY": True})
    for split_name, component in split_jobs:
        split_dir = out / "runs" / split_name
        rows = run_condition(searcher, confirm, queries, qrels, component=component, out_jsonl=split_dir / f"{split_name}.jsonl")
        condition_rows[split_name] = rows
        completed.append(split_name)
        write_status_live(out / "STATUS_LIVE.md", stage="h100_1_contribution_confirm", run_id=manifest["run_id"], n_expected=1 + len(H1001_COMPONENTS) + len(CONTROL_COMPONENTS), n_finished=len(completed), errors=[], extra={"last_split": split_name, "LOCAL_COMPAT_ONLY": True})

    reports = {}
    for component in H1001_COMPONENTS:
        reports[component] = contribution_report(component, condition_rows["full"], condition_rows[f"minus_{component}"], n_boot=args.n_boot, seed=args.seed)

    rows = []
    for component, report in reports.items():
        row = {"component": component, "n": args.n, "quality_positive": report["quality_positive"]}
        for metric, stat in report["metrics"].items():
            row[f"delta_{metric}"] = stat["mean_delta"]
            row[f"{metric}_wlt"] = [stat["win"], stat["loss"], stat["tie"]]
            row[f"{metric}_ci95"] = stat["bootstrap_ci_95"]
        rows.append(row)

    (out / "CONTRIBUTION_CONFIRM.json").write_text(json.dumps({"backend": "local_bm25_compat", "LOCAL_COMPAT_ONLY": True, "confirm_split": "BCP_CONFIRM400", "reports": reports, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row})
    with (out / "CONTRIBUTION_CONFIRM.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = ["# CONTRIBUTION_CONFIRM", "", "Backend: `local_bm25_compat` (`LOCAL_COMPAT_ONLY`).", "", "| component | n | delta curated | delta trajectory | delta final | delta reward | delta context tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['component']} | {row['n']} | {row.get('delta_curated_recall', 0):.6f} | {row.get('delta_trajectory_recall', 0):.6f} | {row.get('delta_final_answer_recall', 0):.6f} | {row.get('delta_harness_reward', 0):.6f} | {row.get('delta_context_tokens', 0):.3f} |")
    (out / "CONTRIBUTION_CONFIRM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    runtime_rows = []
    for name, component in [("full", None), *( (f"minus_{c}", c) for c in H1001_COMPONENTS )]:
        s = summarize_condition(condition_rows[name])
        runtime_rows.append({"split": name, **s})
    with (out / "RUNTIME_COST_CONFIRM.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(runtime_rows[0]))
        writer.writeheader()
        writer.writerows(runtime_rows)

    final_manifest = finalize_run_manifest(manifest, exit_code=0, completed_shards=completed)
    write_run_manifest(out / "RUN_MANIFEST.json", final_manifest)
    write_status_live(out / "STATUS_LIVE.md", stage="h100_1_contribution_confirm", run_id=manifest["run_id"], n_expected=len(completed), n_finished=len(completed), errors=[], extra={"phase": "quality-complete", "LOCAL_COMPAT_ONLY": True})
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    print(json.dumps({"out_dir": str(out), "confirm_n": len(confirm), "rows": len(rows), "LOCAL_COMPAT_ONLY": True}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
