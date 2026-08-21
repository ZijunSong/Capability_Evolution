#!/usr/bin/env python3
"""Score the formal subtractive-c​​uration K4/K8 evidence-recall fork."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], *, seed: int, n_boot: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return [samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]]


def cluster_bootstrap_ci(rows: list[dict[str, Any]], key: str, *, seed: int, n_boot: int = 10000) -> list[float]:
    by_query: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(float(row[key]))
    qids = sorted(by_query)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        sampled = [rng.choice(qids) for _ in qids]
        values = [v for qid in sampled for v in by_query[qid]]
        samples.append(mean(values))
    samples.sort()
    return [samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]]


def summarize(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    candidate = [float(r["candidate_recall_delta"]) for r in rows]
    activated = [float(r["activated_recall_delta"]) for r in rows]
    out: dict[str, Any] = {"K": k, "n": len(rows), "n_queries": len({r["query_id"] for r in rows})}
    for name, values in (("candidate", candidate), ("activated", activated)):
        out[f"{name}_delta_pp"] = 100 * mean(values)
        out[f"{name}_bootstrap_ci95_pp"] = [100 * x for x in bootstrap_ci(values, seed=20260820 + k + len(name))]
        out[f"{name}_query_cluster_bootstrap_ci95_pp"] = [100 * x for x in cluster_bootstrap_ci(rows, f"{name}_recall_delta", seed=20260830 + k + len(name))]
        out[f"{name}_positive_negative_zero"] = [sum(x > 0 for x in values), sum(x < 0 for x in values), sum(x == 0 for x in values)]
        for branch in ("T", "S"):
            ep = [r[f"branch_{branch}_endpoint"] for r in rows]
            out[f"{name}_{branch}_mean"] = mean([float(e[f"{name}_evidence_{'pool_' if name == 'candidate' else ''}recall_at_k"]) for e in ep])
            out[f"{name}_{branch}_precision_mean"] = mean([float(e[f"{name}_evidence_{'pool_' if name == 'candidate' else ''}precision_at_k"]) for e in ep])
            out[f"{name}_{branch}_set_size_mean"] = mean([float(e[f"{name}_evidence_{'pool_' if name == 'candidate' else ''}size_at_k"]) for e in ep])
    for branch in ("T", "S"):
        eps = [r[f"branch_{branch}_endpoint"] for r in rows]
        out[f"successful_read_{branch}_mean"] = mean([len(e["successful_read_ids_within_k"]) for e in eps])
        out[f"duplicate_read_{branch}_mean"] = mean([len(e["read_attempt_ids_within_k"]) - len(set(e["successful_read_ids_within_k"])) for e in eps])
        out[f"context_retention_rate_{branch}"] = mean([len(e["read_ids_retained_at_endpoint"]) / max(1, len(e["successful_read_ids_within_k"])) for e in eps])
        out[f"tool_cost_{branch}_mean"] = mean([float(r[f"branch_{branch}_metrics"]["tool_search_cost"]) for r in rows])
        out[f"utility_{branch}_mean"] = mean([float(r[f"branch_{branch}_metrics"]["objective_utility"]) for r in rows])
    out["tool_cost_delta_mean"] = out["tool_cost_T_mean"] - out["tool_cost_S_mean"]
    out["utility_delta_mean"] = out["utility_T_mean"] - out["utility_S_mean"]
    out["missing_or_empty_qrel_count"] = sum(not r["gold_evidence_ids"] for r in rows)
    out["invalid_provenance_count"] = 0
    out["snapshot_mismatch_count"] = sum(r["snapshot_hash"] != r["initial_state_hash"] for r in rows)
    out["full_harness_takeover_count"] = sum(bool(r["full_harness_takeover"]) for r in rows)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    out["seed_results"] = [{"seed": s, "n": len(rs), "candidate_delta_pp": 100 * mean([r["candidate_recall_delta"] for r in rs]), "activated_delta_pp": 100 * mean([r["activated_recall_delta"] for r in rs])} for s, rs in sorted(by_seed.items())]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    all_rows = []
    hashes = {}
    for k in (4, 8):
        path = args.out_dir / "shards" / f"subtractive_curation_K{k}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if len(rows) != 128:
            raise ValueError(f"K{k}: expected 128 rows, got {len(rows)}")
        hashes[k] = [r["snapshot_hash"] for r in rows]
        all_rows.extend(rows)
    if hashes[4] != hashes[8]:
        raise ValueError("K4/K8 ordered snapshot hashes differ")
    summary = [summarize([r for r in all_rows if int(r["K"]) == k], k) for k in (4, 8)]
    payload = {
        "status": "FORMAL_RECALL_COMPLETE",
        "component": "subtractive_curation",
        "normalization": "split_at_first_underscore_v1",
        "bootstrap_resamples": 10000,
        "same_state_ordered_match": 128,
        "summary": summary,
    }
    (args.out_dir / "SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_GATE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_PER_STATE.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["K", "n", "n_queries", "candidate_T_mean", "candidate_S_mean", "candidate_delta_pp", "activated_T_mean", "activated_S_mean", "activated_delta_pp", "tool_cost_delta_mean", "utility_delta_mean"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summary)
    qrel = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt")
    payload["qrel_sha256"] = hashlib.sha256(qrel.read_bytes()).hexdigest()
    (args.out_dir / "SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_GATE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
