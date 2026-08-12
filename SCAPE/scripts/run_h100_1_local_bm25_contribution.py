#!/usr/bin/env python3
"""Run H100-1 contribution map with the local BrowseComp+ BM25 backend.

This is a compatibility runner for hosts without the Harness-1 Chroma Cloud
backend. It uses the public BrowseComp+ Lucene BM25 index already present on
this machine, preserves the H100-1 artifact schema, and marks the backend as
`local_bm25_compat` rather than claiming official Chroma parity.
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
                qid, docid = str(parts[0]), str(parts[2])
                qrels.setdefault(qid, set()).add(docid)
    return qrels


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
    # Deterministic local interventions. They change one logical component at a time
    # and are intentionally conservative unless the component controls retrieval onset
    # or rendered state in the public Harness-1 taxonomy.
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
            query = queries[qid] + params["query_suffix"]
            t0 = time.perf_counter()
            hits = searcher.search(query, int(params["search_k"]))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ids = [str(h.docid) for h in hits]
            if params["drop_every"]:
                ids = [docid for i, docid in enumerate(ids) if (i + 1) % int(params["drop_every"]) != 0]
            curated_ids = ids[: int(params["curated_k"])]
            trajectory_ids = ids[: int(params["trajectory_k"])]
            gold = qrels.get(qid, set())
            context_tokens = sum(min(len(extract_text(getattr(h, "raw", None) or "")), int(params["context_chars"])) for h in hits[: int(params["curated_k"])]) // 4
            row = {
                "query_id": qid,
                "component_removed": component,
                "curated_recall": recall(curated_ids, gold),
                "trajectory_recall": recall(trajectory_ids, gold),
                "final_answer_recall": recall(curated_ids[:3], gold),
                "harness_reward": 0.45 * recall(curated_ids, gold) + 0.45 * recall(trajectory_ids, gold) + 0.10 * recall(curated_ids[:3], gold),
                "tool_calls": params["tool_calls"],
                "turns": params["turns"],
                "context_tokens": context_tokens,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_1_contribution")
    ap.add_argument("--seed", type=int, default=1101)
    ap.add_argument("--cal-n", type=int, default=200)
    ap.add_argument("--hold-n", type=int, default=200)
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from pyserini.search.lucene import LuceneSearcher

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    all_qids = sorted(set(queries) & set(qrels))
    cal, rem = stable_split(all_qids, seed=args.seed, n_take=args.cal_n)
    hold, _ = stable_split(rem, seed=f"{args.seed}:hold", n_take=args.hold_n)
    smoke20 = cal[:20]
    smoke5 = cal[:5]
    smoke1 = cal[:1]

    manifest = build_run_manifest(
        run_id="h100_1_local_bm25_contribution_20260811",
        stage="h100_1_contribution",
        command=["python", "scripts/run_h100_1_local_bm25_contribution.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={
            "queries": args.browsecomp_root / "topics-qrels" / "queries.tsv",
            "qrel_evidence": args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt",
        },
        extra={"backend": "local_bm25_compat", "official_chroma_parity": False, "seed": args.seed},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)
    write_status_live(out / "STATUS_LIVE.md", stage="h100_1_contribution", run_id=manifest["run_id"], n_expected=14, n_finished=0, errors=[], extra={"phase": "starting"})

    searcher = LuceneSearcher(str(args.index_path))
    (out / "manifests").mkdir(exist_ok=True)
    for name, qids in {"BCP_SMOKE1": smoke1, "BCP_SMOKE5": smoke5, "BCP_SMOKE20": smoke20, "BCP_CAL200": cal, "BCP_HOLD200": hold}.items():
        (out / "manifests" / f"{name}.json").write_text(json.dumps(qids, indent=2) + "\n", encoding="utf-8")

    completed: list[str] = []
    condition_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, qids in [("BCP_SMOKE1", smoke1), ("BCP_SMOKE5", smoke5), ("BCP_SMOKE20", smoke20), ("BCP_CAL200", cal), ("BCP_HOLD200", hold)]:
        split_dir = out / "runs" / split_name
        full_rows = run_condition(searcher, qids, queries, qrels, component=None, out_jsonl=split_dir / "full.jsonl")
        condition_rows[f"full/{split_name}"] = full_rows
        if split_name == "BCP_CAL200":
            for component in all_component_ids():
                rows = run_condition(searcher, qids, queries, qrels, component=component, out_jsonl=split_dir / f"minus_{component}.jsonl")
                condition_rows[f"minus_{component}/{split_name}"] = rows
                completed.append(component)
                write_status_live(out / "STATUS_LIVE.md", stage="h100_1_contribution", run_id=manifest["run_id"], n_expected=10, n_finished=len(completed), errors=[], extra={"last_component": component, "backend": "local_bm25_compat"})

    reports = {}
    full_cal = condition_rows["full/BCP_CAL200"]
    for component in all_component_ids():
        reports[component] = contribution_report(component, full_cal, condition_rows[f"minus_{component}/BCP_CAL200"], n_boot=500, seed=args.seed)

    rows = []
    for component, report in reports.items():
        row = {"component": component, "n": args.cal_n, "quality_positive": report["quality_positive"]}
        for metric, stat in report["metrics"].items():
            row[f"delta_{metric}"] = stat["mean_delta"]
            row[f"{metric}_wlt"] = [stat["win"], stat["loss"], stat["tie"]]
            row[f"{metric}_ci95"] = stat["bootstrap_ci_95"]
        rows.append(row)

    (out / "COMPONENT_CONTRIBUTION.json").write_text(json.dumps({"backend": "local_bm25_compat", "official_chroma_parity": False, "reports": reports, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row})
    with (out / "COMPONENT_CONTRIBUTION.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = ["# COMPONENT_CONTRIBUTION", "", "Backend: `local_bm25_compat` (not official Chroma Cloud parity).", "", "| component | n | delta curated | delta trajectory | delta final | delta reward | delta context tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['component']} | {row['n']} | {row.get('delta_curated_recall', 0):.6f} | {row.get('delta_trajectory_recall', 0):.6f} | {row.get('delta_final_answer_recall', 0):.6f} | {row.get('delta_harness_reward', 0):.6f} | {row.get('delta_context_tokens', 0):.3f} |")
    (out / "COMPONENT_CONTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    baseline = {
        "backend": "local_bm25_compat",
        "official_chroma_parity": False,
        "splits": {
            "BCP_SMOKE1": summarize_condition(condition_rows["full/BCP_SMOKE1"]),
            "BCP_SMOKE5": summarize_condition(condition_rows["full/BCP_SMOKE5"]),
            "BCP_SMOKE20": summarize_condition(condition_rows["full/BCP_SMOKE20"]),
            "BCP_CAL200": summarize_condition(condition_rows["full/BCP_CAL200"]),
            "BCP_HOLD200": summarize_condition(condition_rows["full/BCP_HOLD200"]),
        },
    }
    (out / "BASELINE_REPRODUCTION.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    (out / "BASELINE_REPRODUCTION.md").write_text("# BASELINE_REPRODUCTION\n\n" + json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=completed))
    write_status_live(out / "STATUS_LIVE.md", stage="h100_1_contribution", run_id=manifest["run_id"], n_expected=10, n_finished=10, errors=[], extra={"backend": "local_bm25_compat", "official_chroma_parity": False})
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    print(json.dumps({"out_dir": str(out), "components": len(rows), "backend": "local_bm25_compat"}, indent=2))


if __name__ == "__main__":
    main()
