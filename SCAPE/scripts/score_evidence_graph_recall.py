#!/usr/bin/env python3
"""Independently score evidence_graph endpoint evidence recall forks."""
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                qrels[str(parts[0])].add(str(parts[2]))
    return dict(qrels)


def norm(values: list[str] | set[str]) -> set[str]:
    return {str(value).split("_", 1)[0] for value in values if str(value)}


def recall(values: list[str], gold: set[str]) -> float:
    ngold = norm(gold)
    return len(norm(values) & ngold) / len(ngold) if ngold else 0.0


def precision(values: list[str], gold: set[str]) -> float:
    nvalues = norm(values)
    return len(nvalues & norm(gold)) / len(nvalues) if nvalues else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_ci(values: list[float], seed: int, n_boot: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return [draws[int(0.025 * n_boot)], draws[min(n_boot - 1, int(0.975 * n_boot))]]


def cluster_bootstrap_ci(rows: list[dict[str, Any]], field: str, seed: int, n_boot: int) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(float(row[field]))
    keys = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(mean([value for key in sampled for value in grouped[key]]))
    draws.sort()
    return [draws[int(0.025 * n_boot)], draws[min(n_boot - 1, int(0.975 * n_boot))]]


def score_row(row: dict[str, Any], qrels: dict[str, set[str]]) -> dict[str, Any]:
    qid = str(row["query_id"])
    gold = qrels.get(qid, set())
    student = row["branch_S_endpoint"]
    teacher = row["branch_T_endpoint"]
    for label, endpoint in (("student", student), ("teacher", teacher)):
        expected = norm(endpoint["final_curated_ids"]) | norm(endpoint["read_ids_retained_at_endpoint"])
        if norm(endpoint["final_activated_evidence_ids"]) != expected:
            raise ValueError(f"{row['state_id']} {label}: activated union mismatch")
        if not norm(endpoint["read_ids_retained_at_endpoint"]).issubset(norm(endpoint["successful_read_ids_within_k"])):
            raise ValueError(f"{row['state_id']} {label}: retained read lacks success")
        if norm(endpoint["successful_read_ids_within_k"]) != norm(endpoint["read_ids_entered_context"]):
            raise ValueError(f"{row['state_id']} {label}: read/context mismatch")
    result = {
        "seed": int(row["seed"]),
        "K": int(row["K"]),
        "state_id": row["state_id"],
        "query_id": qid,
        "turn_id": int(row["turn_id"]),
        "snapshot_hash": row["snapshot_hash"],
        "gold_evidence_ids": sorted(gold),
        "student_candidate_ids": student["final_candidate_evidence_ids"],
        "teacher_candidate_ids": teacher["final_candidate_evidence_ids"],
        "student_activated_ids": student["final_activated_evidence_ids"],
        "teacher_activated_ids": teacher["final_activated_evidence_ids"],
        "student_candidate_recall": recall(student["final_candidate_evidence_ids"], gold),
        "teacher_candidate_recall": recall(teacher["final_candidate_evidence_ids"], gold),
        "student_candidate_precision": precision(student["final_candidate_evidence_ids"], gold),
        "teacher_candidate_precision": precision(teacher["final_candidate_evidence_ids"], gold),
        "student_candidate_size": len(norm(student["final_candidate_evidence_ids"])),
        "teacher_candidate_size": len(norm(teacher["final_candidate_evidence_ids"])),
        "student_activated_recall": recall(student["final_activated_evidence_ids"], gold),
        "teacher_activated_recall": recall(teacher["final_activated_evidence_ids"], gold),
        "student_activated_precision": precision(student["final_activated_evidence_ids"], gold),
        "teacher_activated_precision": precision(teacher["final_activated_evidence_ids"], gold),
        "student_activated_size": len(norm(student["final_activated_evidence_ids"])),
        "teacher_activated_size": len(norm(teacher["final_activated_evidence_ids"])),
        "student_successful_read_count": len(norm(student["successful_read_ids_within_k"])),
        "teacher_successful_read_count": len(norm(teacher["successful_read_ids_within_k"])),
        "student_duplicate_read_count": max(0, len(student["read_attempt_ids_within_k"]) - len(norm(student["read_attempt_ids_within_k"]))),
        "teacher_duplicate_read_count": max(0, len(teacher["read_attempt_ids_within_k"]) - len(norm(teacher["read_attempt_ids_within_k"]))),
        "student_context_retention_rate": len(norm(student["read_ids_retained_at_endpoint"])) / max(1, len(norm(student["successful_read_ids_within_k"]))),
        "teacher_context_retention_rate": len(norm(teacher["read_ids_retained_at_endpoint"])) / max(1, len(norm(teacher["successful_read_ids_within_k"]))),
        "student_curate_add_count": sum(len(step["action"].get("arguments", {}).get("add_ids", [])) for step in row["branch_S_trace"] if step["action"].get("name") == "curate"),
        "teacher_curate_add_count": sum(len(step["action"].get("arguments", {}).get("add_ids", [])) for step in row["branch_T_trace"] if step["action"].get("name") == "curate"),
        "student_curate_remove_count": sum(len(step["action"].get("arguments", {}).get("remove_ids", [])) for step in row["branch_S_trace"] if step["action"].get("name") == "curate"),
        "teacher_curate_remove_count": sum(len(step["action"].get("arguments", {}).get("remove_ids", [])) for step in row["branch_T_trace"] if step["action"].get("name") == "curate"),
        "student_tool_cost": float(row["branch_S_metrics"]["tool_search_cost"]),
        "teacher_tool_cost": float(row["branch_T_metrics"]["tool_search_cost"]),
        "student_utility": float(row["branch_S_metrics"]["objective_utility"]),
        "teacher_utility": float(row["branch_T_metrics"]["objective_utility"]),
        "full_harness_takeover": bool(row["full_harness_takeover"]),
    }
    result["candidate_delta"] = result["teacher_candidate_recall"] - result["student_candidate_recall"]
    result["activated_delta"] = result["teacher_activated_recall"] - result["student_activated_recall"]
    result["tool_cost_delta"] = result["teacher_tool_cost"] - result["student_tool_cost"]
    result["utility_delta"] = result["teacher_utility"] - result["student_utility"]
    return result


def aggregate(rows: list[dict[str, Any]], field: str, seed: int, n_boot: int) -> dict[str, Any]:
    deltas = [float(row[field]) for row in rows]
    teacher_field = "teacher_" + field.removesuffix("_delta") + "_recall"
    student_field = "student_" + field.removesuffix("_delta") + "_recall"
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(float(row[field]))
    seed_means = {str(key): mean(values) for key, values in sorted(by_seed.items())}
    return {
        "n": len(rows),
        "teacher_mean": mean([float(row[teacher_field]) for row in rows]),
        "student_mean": mean([float(row[student_field]) for row in rows]),
        "paired_mean_delta": mean(deltas),
        "paired_mean_delta_pp": 100.0 * mean(deltas),
        "paired_row_bootstrap_ci95_pp": [100.0 * value for value in bootstrap_ci(deltas, seed, n_boot)],
        "query_cluster_bootstrap_ci95_pp": [100.0 * value for value in cluster_bootstrap_ci(rows, field, seed + 1, n_boot)],
        "positive_negative_zero": [sum(value > 0 for value in deltas), sum(value < 0 for value in deltas), sum(value == 0 for value in deltas)],
        "per_seed_delta_pp": {key: 100.0 * value for key, value in seed_means.items()},
        "seed_mean_delta_pp": 100.0 * mean(list(seed_means.values())),
        "seed_sample_std_pp": 100.0 * statistics.stdev(seed_means.values()) if len(seed_means) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--states-cache", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-states", type=int, default=128)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = [row for path in args.input for row in load_jsonl(path)]
    frozen_states = load_jsonl(args.states_cache)
    cache_index = {row["snapshot_hash"]: index for index, row in enumerate(frozen_states)}
    if len(cache_index) != args.n_states:
        raise ValueError(f"states cache has {len(cache_index)} unique hashes, expected {args.n_states}")
    for row in raw:
        if row["snapshot_hash"] not in cache_index:
            raise ValueError(f"result snapshot absent from frozen cache: {row['snapshot_hash']}")
        row["state_id"] = f"evidence_graph_K{row['K']}_{cache_index[row['snapshot_hash']]:03d}"
    qrels = load_qrels(args.qrels)
    scored = [score_row(row, qrels) for row in raw]
    audits: dict[str, Any] = {"invalid_provenance": 0, "snapshot_mismatch": 0, "full_harness_takeover": sum(row["full_harness_takeover"] for row in scored)}
    by_k = {k: sorted((row for row in scored if row["K"] == k), key=lambda row: row["state_id"]) for k in (4, 8)}
    for k, rows in by_k.items():
        if len(rows) != args.n_states or len({row["snapshot_hash"] for row in rows}) != args.n_states:
            raise ValueError(f"K{k}: expected {args.n_states} unique rows, got {len(rows)}")
    k4_hashes = [row["snapshot_hash"] for row in by_k[4]]
    k8_hashes = [row["snapshot_hash"] for row in by_k[8]]
    audits["ordered_snapshot_match"] = sum(left == right for left, right in zip(k4_hashes, k8_hashes))
    if k4_hashes != k8_hashes:
        audits["snapshot_mismatch"] = sum(left != right for left, right in zip(k4_hashes, k8_hashes))
        raise ValueError("K4/K8 ordered snapshot hashes differ")
    summaries = {}
    for k, rows in by_k.items():
        candidate = aggregate(rows, "candidate_delta", args.bootstrap_seed + k * 10, args.n_bootstrap)
        activated = aggregate(rows, "activated_delta", args.bootstrap_seed + k * 10 + 2, args.n_bootstrap)
        summaries[f"K{k}"] = {
            "candidate_evidence_pool_recall": candidate,
            "activated_evidence_recall": activated,
            "candidate_precision_teacher_student": [mean([row["teacher_candidate_precision"] for row in rows]), mean([row["student_candidate_precision"] for row in rows])],
            "candidate_size_teacher_student": [mean([row["teacher_candidate_size"] for row in rows]), mean([row["student_candidate_size"] for row in rows])],
            "activated_precision_teacher_student": [mean([row["teacher_activated_precision"] for row in rows]), mean([row["student_activated_precision"] for row in rows])],
            "activated_size_teacher_student": [mean([row["teacher_activated_size"] for row in rows]), mean([row["student_activated_size"] for row in rows])],
            "successful_reads_teacher_student": [mean([row["teacher_successful_read_count"] for row in rows]), mean([row["student_successful_read_count"] for row in rows])],
            "duplicate_reads_teacher_student": [mean([row["teacher_duplicate_read_count"] for row in rows]), mean([row["student_duplicate_read_count"] for row in rows])],
            "context_retention_teacher_student": [
                sum(row["teacher_successful_read_count"] * row["teacher_context_retention_rate"] for row in rows) / max(1, sum(row["teacher_successful_read_count"] for row in rows)),
                sum(row["student_successful_read_count"] * row["student_context_retention_rate"] for row in rows) / max(1, sum(row["student_successful_read_count"] for row in rows)),
            ],
            "curate_add_teacher_student": [mean([row["teacher_curate_add_count"] for row in rows]), mean([row["student_curate_add_count"] for row in rows])],
            "curate_remove_teacher_student": [mean([row["teacher_curate_remove_count"] for row in rows]), mean([row["student_curate_remove_count"] for row in rows])],
            "tool_cost_delta": mean([row["tool_cost_delta"] for row in rows]),
            "weighted_utility_delta": mean([row["utility_delta"] for row in rows]),
            "missing_or_empty_qrel_count": sum(not row["gold_evidence_ids"] for row in rows),
        }
    payload = {
        "component": "evidence_graph",
        "contract": "Teacher component-on first action vs Student component-off first action; first action counts toward K; both continuations reduced",
        "normalization": "split_at_first_underscore_v1",
        "context_retention_policy": "successful_reads_append_only_retained_to_endpoint",
        "qrel_path": str(args.qrels),
        "qrel_sha256": sha256(args.qrels),
        "states_cache": str(args.states_cache),
        "states_cache_sha256": sha256(args.states_cache),
        "runner_sha256": sha256(args.runner),
        "scorer_sha256": sha256(Path(__file__)),
        "audits": audits,
        "horizons": summaries,
    }
    with (args.out_dir / "EVIDENCE_GRAPH_EVIDENCE_RECALL_PER_STATE.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(scored, key=lambda item: (item["K"], item["state_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "EVIDENCE_GRAPH_EVIDENCE_RECALL_SUMMARY.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(args.out_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
