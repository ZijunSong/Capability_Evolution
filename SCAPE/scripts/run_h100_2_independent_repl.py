#!/usr/bin/env python3
"""Run H100-2 independent 10-component replication on BCP_REPL200_V2.

This is the 2026-08-12 H100-2 fresh split runner. It intentionally uses the
SCAPE local BM25 compatibility path and marks every artifact LOCAL_COMPAT_ONLY;
it does not consume the older SCOPE h100_2_module_utility consolidation as an
experimental input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import all_component_ids
from scape.common.hashing import stable_split
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.eval.paired_bootstrap import paired_query_stats

COMPONENTS = all_component_ids()
COALITIONS = {
    "K1_evidence_graph_subtractive_curation": ["evidence_graph", "subtractive_curation"],
    "K2_importance_tagging_subtractive_curation": ["importance_tagging", "subtractive_curation"],
    "K3_evidence_graph_chunk_neighbors": ["evidence_graph", "chunk_neighbors"],
    "K4_auto_populate_first_search_adaptive_rerank_instruction": ["auto_populate_first_search", "adaptive_rerank_instruction"],
}
METRICS = (
    "curated_recall",
    "trajectory_recall",
    "final_answer_recall",
    "harness_reward",
    "turns",
    "tool_calls",
    "context_tokens",
)


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
            raw = obj.get("query_ids") or obj.get("qids") or obj.get("BCP_CONFIRM400") or obj.get("BCP_CAL200") or []
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


def condition_params(disabled: list[str]) -> dict[str, Any]:
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
    for component in disabled:
        if component == "auto_populate_first_search":
            params.update(search_k=25, trajectory_k=25, tool_calls=3, turns=5, state_ops=6)
        elif component == "subtractive_curation":
            params.update(curated_k=min(params["curated_k"], 6), state_ops=min(params["state_ops"], 5))
        elif component == "importance_tagging":
            params.update(curated_k=min(params["curated_k"], 8), query_suffix=(params["query_suffix"] + " evidence").strip())
        elif component == "evidence_graph":
            params.update(trajectory_k=min(params["trajectory_k"], 35), state_ops=min(params["state_ops"], 5))
        elif component == "sentence_compress":
            params.update(context_chars=min(params["context_chars"], 600))
        elif component == "content_dedup":
            params.update(drop_every=5, context_chars=max(params["context_chars"], 1600), state_ops=max(params["state_ops"], 11))
        elif component == "chunk_neighbors":
            params.update(search_k=min(params["search_k"], 35), trajectory_k=min(params["trajectory_k"], 35))
        elif component == "verify_tool":
            params.update(tool_calls=min(params["tool_calls"], 3), turns=4, state_ops=min(params["state_ops"], 6))
        elif component == "token_budget_marker":
            params.update(context_chars=max(params["context_chars"], 1800), turns=max(params["turns"], 5))
        elif component == "adaptive_rerank_instruction":
            params.update(query_suffix=(params["query_suffix"] + " relevant source").strip())
    return params


def recall(hit_ids: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    norm_hits = {str(x).split("_", 1)[0] for x in hit_ids}
    return len(norm_hits & gold) / len(gold)


def run_condition(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from pyserini.search.lucene import LuceneSearcher

    name = task["name"]
    disabled = task["disabled"]
    qids = task["qids"]
    queries = load_queries(Path(task["queries_path"]))
    qrels = load_qrels(Path(task["qrels_path"]))
    searcher = LuceneSearcher(str(task["index_path"]))
    params = condition_params(disabled)
    out_jsonl = Path(task["out_jsonl"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    with out_jsonl.open("w", encoding="utf-8") as f:
        for qid in qids:
            query = queries[qid]
            if params["query_suffix"]:
                query = f"{query} {params['query_suffix']}"
            t0 = time.perf_counter()
            hits = searcher.search(query, int(params["search_k"]))
            latency_ms = (time.perf_counter() - t0) * 1000
            ids = [str(h.docid) for h in hits]
            if params["drop_every"]:
                ids = [docid for i, docid in enumerate(ids) if (i + 1) % int(params["drop_every"]) != 0]
            curated_ids = ids[: int(params["curated_k"])]
            trajectory_ids = ids[: int(params["trajectory_k"])]
            gold = qrels.get(qid, set())
            curated_recall = recall(curated_ids, gold)
            trajectory_recall = recall(trajectory_ids, gold)
            final_recall = recall(curated_ids[:3], gold)
            context_tokens = sum(
                min(len(extract_text(getattr(h, "raw", None) or "")), int(params["context_chars"]))
                for h in hits[: int(params["curated_k"])]
            ) // 4
            row = {
                "query_id": qid,
                "condition": name,
                "disabled_components": disabled,
                "curated_recall": curated_recall,
                "trajectory_recall": trajectory_recall,
                "final_answer_recall": final_recall,
                "harness_reward": 0.45 * curated_recall + 0.45 * trajectory_recall + 0.10 * final_recall,
                "tool_calls": params["tool_calls"],
                "turns": params["turns"],
                "context_tokens": context_tokens,
                "latency_ms": latency_ms,
                "state_ops": params["state_ops"],
                "retrieved_docids": ids[:50],
                "gold_docids": sorted(gold),
                "backend": "local_bm25_compat",
                "LOCAL_COMPAT_ONLY": True,
                "error": False,
            }
            rows[qid] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"name": name, "rows": rows, "out_jsonl": str(out_jsonl), "n": len(rows)}


def summarize(full: dict[str, Any], other: dict[str, Any], *, label: str, disabled: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {"condition": label, "disabled_components": disabled, "n": len(set(full) & set(other))}
    for metric in METRICS:
        stats = paired_query_stats(other, full, metric=metric, n_boot=n_boot, seed=seed)
        row[f"delta_{metric}"] = stats["mean_delta"]
        row[f"{metric}_wlt"] = f"{stats['win']}/{stats['loss']}/{stats['tie']}"
        row[f"{metric}_ci95"] = stats["bootstrap_ci_95"]
    row["quality_positive"] = any(float(row.get(f"delta_{m}", 0.0)) > 0 for m in ("curated_recall", "trajectory_recall", "final_answer_recall", "harness_reward"))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_2_independent_repl")
    ap.add_argument("--seed", type=int, default=2203)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    queries_path = args.browsecomp_root / "topics-qrels" / "queries.tsv"
    qrels_path = args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)

    manifest = build_run_manifest(
        run_id="h100_2_independent_repl_20260812",
        stage="h100_2_independent_repl",
        command=["python", "scripts/run_h100_2_independent_repl.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"queries": queries_path, "qrel_evidence": qrels_path},
        extra={"backend": "local_bm25_compat", "LOCAL_COMPAT_ONLY": True, "seed": args.seed, "n": args.n},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    all_qids = sorted(set(queries) & set(qrels))
    exclusions = load_existing_qids([
        REPO / "outputs" / "h100_1_contribution" / "manifests" / "BCP_CAL200.json",
        REPO / "outputs" / "h100_1_contribution" / "manifests" / "BCP_HOLD200.json",
        REPO / "outputs" / "h100_1_contribution_confirm" / "manifests" / "BCP_CONFIRM400.json",
        REPO / "outputs" / "h100_1_contribution_confirm" / "manifests" / "BCP_CAL200.json",
    ])
    excluded = {qid for qids in exclusions.values() for qid in qids}
    eligible = [qid for qid in all_qids if qid not in excluded]
    repl, _ = stable_split(eligible, seed=args.seed, n_take=args.n)
    if len(repl) < args.n:
        raise RuntimeError(f"Only {len(repl)} eligible qids for requested n={args.n}")
    (out / "manifests" / "bcp_repl200_v2.json").write_text(json.dumps({"name": "BCP_REPL200_V2", "seed": args.seed, "n": len(repl), "query_ids": repl}, indent=2) + "\n", encoding="utf-8")
    (out / "manifests" / "bcp_repl200_v2_query_ids.json").write_text(json.dumps(repl, indent=2) + "\n", encoding="utf-8")
    audit_lines = [
        "# REPLICATION_SPLIT_AUDIT",
        "",
        "- split: `BCP_REPL200_V2`",
        f"- seed: {args.seed}",
        f"- n: {len(repl)}",
        f"- all_qids_with_qrels: {len(all_qids)}",
        f"- excluded_qids_from_existing_manifests: {len(excluded)}",
        f"- eligible_after_exclusion: {len(eligible)}",
        "- backend: `local_bm25_compat` / `LOCAL_COMPAT_ONLY`",
        "",
        "## Exclusion Sources",
    ]
    for path, qids in exclusions.items():
        audit_lines.append(f"- `{path}`: {len(qids)} qids")
    if not exclusions:
        audit_lines.append("- No H100-1 manifest found; stable-hash seed selection used.")
    (out / "manifests" / "REPLICATION_SPLIT_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    tasks = [{"name": "R0_full", "disabled": [], "qids": repl, "out_jsonl": str(out / "runs" / "R0_full.jsonl")}] + [
        {"name": f"R{i}_minus_{component}", "disabled": [component], "qids": repl, "out_jsonl": str(out / "runs" / f"R{i}_minus_{component}.jsonl")}
        for i, component in enumerate(COMPONENTS, start=1)
    ]
    completed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    write_status_live(out / "STATUS_LIVE.md", stage="h100_2_independent_repl", run_id=manifest["run_id"], n_expected=16, n_finished=0, errors=[], extra={"phase": "running full + 10 LOO", "LOCAL_COMPAT_ONLY": True})
    common = {"queries_path": str(queries_path), "qrels_path": str(qrels_path), "index_path": str(args.index_path)}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_condition, {**task, **common}): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                result = fut.result()
                completed[result["name"]] = result["rows"]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['name']}: {exc}")
            write_status_live(out / "STATUS_LIVE.md", stage="h100_2_independent_repl", run_id=manifest["run_id"], n_expected=16, n_finished=len(completed), errors=errors, extra={"last_finished": task["name"], "phase": "running full + 10 LOO", "LOCAL_COMPAT_ONLY": True})
    if errors:
        write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=1, error_summary="; ".join(errors[:3]), completed_shards=list(completed)))
        return 1

    full = completed["R0_full"]
    loo_rows = []
    for i, component in enumerate(COMPONENTS, start=1):
        name = f"R{i}_minus_{component}"
        loo_rows.append({"component": component, **summarize(full, completed[name], label=name, disabled=[component], n_boot=args.n_boot, seed=args.seed)})
    write_csv(out / "LOO_REPLICATION_V2.csv", loo_rows)
    (out / "LOO_REPLICATION_V2.json").write_text(json.dumps({"backend": "local_bm25_compat", "LOCAL_COMPAT_ONLY": True, "split": "BCP_REPL200_V2", "rows": loo_rows}, indent=2) + "\n", encoding="utf-8")

    replay_task = {"name": "R0b_full_first40_replay_parity", "disabled": [], "qids": repl[:40], "out_jsonl": str(out / "runs" / "R0b_full_first40_replay_parity.jsonl"), **common}
    replay = run_condition(replay_task)["rows"]
    replay_row = summarize({qid: full[qid] for qid in repl[:40]}, replay, label="R0b_full_first40_replay_parity", disabled=[], n_boot=args.n_boot, seed=args.seed)

    coalition_results = {}
    for name, disabled in COALITIONS.items():
        result = run_condition({"name": name, "disabled": disabled, "qids": repl, "out_jsonl": str(out / "runs" / f"{name}.jsonl"), **common})
        coalition_results[name] = result["rows"]
        write_status_live(out / "STATUS_LIVE.md", stage="h100_2_independent_repl", run_id=manifest["run_id"], n_expected=16, n_finished=12 + len(coalition_results), errors=[], extra={"last_finished": name, "phase": "running coalitions", "LOCAL_COMPAT_ONLY": True})

    coal_rows = []
    loo_by_component = {row["component"]: row for row in loo_rows}
    for name, disabled in COALITIONS.items():
        row = summarize(full, coalition_results[name], label=name, disabled=disabled, n_boot=args.n_boot, seed=args.seed)
        for metric in ("curated_recall", "trajectory_recall", "final_answer_recall", "harness_reward"):
            singles = sum(float(loo_by_component[c].get(f"delta_{metric}", 0.0)) for c in disabled)
            row[f"interaction_{metric}"] = float(row[f"delta_{metric}"]) - singles
        coal_rows.append(row)
    write_csv(out / "COALITION_V2.csv", coal_rows)

    placement = ["# PLACEMENT_STABILITY_V2", "", "Backend: `local_bm25_compat` (`LOCAL_COMPAT_ONLY`).", "", "| component | delta_curated | delta_trajectory | delta_final | delta_reward | quality | placement note |", "|---|---:|---:|---:|---:|---|---|"]
    runtime_like = {"chunk_neighbors", "content_dedup", "token_budget_marker"}
    for row in loo_rows:
        component = row["component"]
        note = "runtime/hybrid control; do not prioritize full internalization" if component in runtime_like else "candidate-eligible pending cross-split and real-influence gates"
        placement.append(f"| {component} | {row['delta_curated_recall']:.6f} | {row['delta_trajectory_recall']:.6f} | {row['delta_final_answer_recall']:.6f} | {row['delta_harness_reward']:.6f} | {row['quality_positive']} | {note} |")
    (out / "PLACEMENT_STABILITY_V2.md").write_text("\n".join(placement) + "\n", encoding="utf-8")

    comparison = ["# CROSS_SPLIT_COMPARISON", "", "This H100-2 run is an independent `BCP_REPL200_V2` fresh split and does not use old SCOPE H100-2 consolidation as input.", "", f"Replay parity first40 max abs delta reward: {abs(float(replay_row['delta_harness_reward'])):.12f}", "", "H100-1 comparison should be regenerated when H100-1 `CONFIRM400` is available; current output is marked `LOCAL_COMPAT_ONLY`."]
    (out / "CROSS_SPLIT_COMPARISON.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")

    final_manifest = finalize_run_manifest(manifest, exit_code=0, completed_shards=["R0_full", *[f"minus_{c}" for c in COMPONENTS], "R0b_first40", *COALITIONS])
    write_run_manifest(out / "RUN_MANIFEST.json", final_manifest)
    write_status_live(out / "STATUS_LIVE.md", stage="h100_2_independent_repl", run_id=manifest["run_id"], n_expected=16, n_finished=16, errors=[], extra={"phase": "quality-complete", "split": "BCP_REPL200_V2", "LOCAL_COMPAT_ONLY": True, "replay_delta_reward": replay_row["delta_harness_reward"]})
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    print(json.dumps({"out_dir": str(out), "split_n": len(repl), "loo": len(loo_rows), "coalitions": len(coal_rows), "LOCAL_COMPAT_ONLY": True}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
