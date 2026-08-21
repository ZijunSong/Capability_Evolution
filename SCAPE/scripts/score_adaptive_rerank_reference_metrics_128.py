#!/usr/bin/env python3
"""Score branch cost and objective utility from formal adaptive recall rows."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from collections import defaultdict
from pathlib import Path

SEEDS = (2214, 2215, 2216, 2217)
KS = (4, 8)

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def objective(metrics, gold_count):
    return (
        0.45 * metrics["evidence_coverage"]
        + 0.20 * metrics["useful_unique_docs"] / max(1, gold_count)
        + 0.20 * metrics["verified_supported_claims"] / max(1, gold_count)
        - 0.05 * metrics["redundancy"]
        - 0.015 * metrics["tool_search_cost"]
        - 0.03 * metrics["unsupported_claims"]
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    errors = []
    source_hashes = {}
    for seed in SEEDS:
        for k in KS:
            path = args.root / "shards" / f"adaptive_rerank_instruction_seed{seed}_K{k}.jsonl"
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    gold_count = len(row.get("gold_evidence_ids", []))
                    sm = row["branch_S_metrics"]
                    tm = row["branch_T_metrics"]
                    for branch, metrics, endpoint in (("S", sm, row["branch_S_endpoint"]), ("T", tm, row["branch_T_endpoint"])):
                        calc = objective(metrics, gold_count)
                        if abs(calc - metrics["objective_utility"]) > 1e-12:
                            errors.append({"path": str(path), "line": line_no, "branch": branch, "kind": "objective_formula", "calculated": calc, "recorded": metrics["objective_utility"]})
                        if abs(float(metrics["tool_search_cost"]) - float(endpoint["tool_cost"])) > 1e-12:
                            errors.append({"path": str(path), "line": line_no, "branch": branch, "kind": "endpoint_cost", "metrics": metrics["tool_search_cost"], "endpoint": endpoint["tool_cost"]})
                    if len(row["branch_S_endpoint"]["actions"]) != int(row["K"]) or len(row["branch_T_endpoint"]["actions"]) != int(row["K"]):
                        errors.append({"path": str(path), "line": line_no, "kind": "horizon_action_count", "K": row["K"], "S": len(row["branch_S_endpoint"]["actions"]), "T": len(row["branch_T_endpoint"]["actions"])})
                    item = {"seed": row["seed"], "K": row["K"], "state_id": row["state_id"], "query_id": row["query_id"], "snapshot_hash": row["snapshot_hash"], "tool_cost_T": tm["tool_search_cost"], "tool_cost_S": sm["tool_search_cost"], "tool_cost_delta": tm["tool_search_cost"] - sm["tool_search_cost"], "utility_T": tm["objective_utility"], "utility_S": sm["objective_utility"], "utility_delta": tm["objective_utility"] - sm["objective_utility"]}
                    rows.append(item)
    summaries = []
    for k in KS:
        rs = [r for r in rows if r["K"] == k]
        summaries.append({"K": k, "n": len(rs), "tool_cost_T_mean": mean([r["tool_cost_T"] for r in rs]), "tool_cost_S_mean": mean([r["tool_cost_S"] for r in rs]), "tool_cost_delta_mean": mean([r["tool_cost_delta"] for r in rs]), "utility_T_mean": mean([r["utility_T"] for r in rs]), "utility_S_mean": mean([r["utility_S"] for r in rs]), "utility_delta_mean": mean([r["utility_delta"] for r in rs]), "tool_cost_delta_positive": sum(r["tool_cost_delta"] > 1e-12 for r in rs), "tool_cost_delta_negative": sum(r["tool_cost_delta"] < -1e-12 for r in rs), "tool_cost_delta_zero": sum(abs(r["tool_cost_delta"]) <= 1e-12 for r in rs), "utility_delta_positive": sum(r["utility_delta"] > 1e-12 for r in rs), "utility_delta_negative": sum(r["utility_delta"] < -1e-12 for r in rs), "utility_delta_zero": sum(abs(r["utility_delta"]) <= 1e-12 for r in rs), "per_seed": {str(seed): {"tool_cost_delta_mean": mean([r["tool_cost_delta"] for r in rs if r["seed"] == seed]), "utility_delta_mean": mean([r["utility_delta"] for r in rs if r["seed"] == seed])} for seed in SEEDS}})
    payload = {"component": "adaptive_rerank_instruction", "source": str(args.root), "rows": len(rows), "summary": summaries, "audit": {"source_sha256": source_hashes, "formula_mismatches": errors, "formula_mismatch_count": len(errors), "formal_recall_rows": True, "forced_action_included_in_K": True, "continuation_policy": "reduced", "full_harness_takeover": False, "normalization": "split_at_first_underscore", "qrel_sha256": "a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1"}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ADAPTIVE_RERANK_COST_UTILITY_PER_STATE.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.out / "ADAPTIVE_RERANK_COST_UTILITY_SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
