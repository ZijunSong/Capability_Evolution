#!/usr/bin/env python3
"""Score token_budget_marker always-on/off paired process metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_action(action: Any) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--states-cache", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-states", type=int, default=128)
    args = parser.parse_args()

    frozen = load_jsonl(args.states_cache)
    if len(frozen) != args.n_states or len({row["snapshot_hash"] for row in frozen}) != args.n_states:
        raise ValueError("frozen state cache is not 128 unique snapshots")
    frozen_hashes = [row["snapshot_hash"] for row in frozen]
    frozen_by_hash = {row["snapshot_hash"]: row for row in frozen}

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_state = []
    audit = {
        "invalid_provenance": 0,
        "snapshot_mismatch": 0,
        "trace_length_mismatch": 0,
        "metric_formula_mismatch": 0,
        "full_harness_takeover": 0,
    }
    for path in args.input:
        for row in load_jsonl(path):
            k = int(row["K"])
            groups[k].append(row)
            if row["snapshot_hash"] not in frozen_by_hash:
                audit["snapshot_mismatch"] += 1
            if row.get("continuation_policy") != "full_teacher_reduced_student" or row.get("teacher_continuation_policy") != "full_always" or row.get("student_continuation_policy") != "reduced_always" or not row.get("frozen_first_actions"):
                audit["invalid_provenance"] += 1
            if row.get("component_mask_first_action") != {"teacher": "full_component_on", "student": "reduced_component_off"}:
                audit["invalid_provenance"] += 1
            if row.get("full_harness_takeover"):
                audit["full_harness_takeover"] += 1
            if len(row["branch_T_trace"]) != k or len(row["branch_S_trace"]) != k:
                audit["trace_length_mismatch"] += 1
            t_cost = float(row["branch_T_metrics"]["tool_search_cost"])
            s_cost = float(row["branch_S_metrics"]["tool_search_cost"])
            t_utility = float(row["branch_T_metrics"]["objective_utility"])
            s_utility = float(row["branch_S_metrics"]["objective_utility"])
            cost_delta = t_cost - s_cost
            utility_delta = t_utility - s_utility
            if abs(cost_delta - float(row["tool_search_cost"])) > 1e-12 or abs(utility_delta - float(row["branch_T_minus_S"])) > 1e-12:
                audit["metric_formula_mismatch"] += 1
            per_state.append({
                "K": k,
                "state_id": row["state_id"],
                "query_id": row["query_id"],
                "snapshot_hash": row["snapshot_hash"],
                "first_action_disagreement": int(canonical_action(row["a_T"]) != canonical_action(row["a_S"])),
                "teacher_tool_cost": t_cost,
                "student_tool_cost": s_cost,
                "tool_cost_delta": cost_delta,
                "teacher_utility": t_utility,
                "student_utility": s_utility,
                "utility_delta": utility_delta,
            })

    horizons = {}
    for k in (4, 8):
        rows = groups[k]
        ordered = [row["snapshot_hash"] for row in rows]
        if len(rows) != args.n_states or len(set(ordered)) != args.n_states:
            raise ValueError(f"K{k} does not have 128 unique rows")
        if ordered != frozen_hashes:
            audit["snapshot_mismatch"] += sum(a != b for a, b in zip(ordered, frozen_hashes))
        scored = [row for row in per_state if row["K"] == k]
        horizons[f"K{k}"] = {
            "n": len(scored),
            "first_action_disagreement_rate": mean(row["first_action_disagreement"] for row in scored),
            "first_action_disagreement_rate_percent": 100 * mean(row["first_action_disagreement"] for row in scored),
            "teacher_tool_cost_mean": mean(row["teacher_tool_cost"] for row in scored),
            "student_tool_cost_mean": mean(row["student_tool_cost"] for row in scored),
            "tool_cost_delta": mean(row["tool_cost_delta"] for row in scored),
            "teacher_utility_mean": mean(row["teacher_utility"] for row in scored),
            "student_utility_mean": mean(row["student_utility"] for row in scored),
            "utility_delta": mean(row["utility_delta"] for row in scored),
            "utility_delta_percent": 100 * mean(row["utility_delta"] for row in scored),
        }

    if any(audit.values()):
        raise ValueError(f"audit failed: {audit}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": "token_budget_marker",
        "contract": "frozen first actions; Teacher full/component-on for every continuation; Student reduced/component-off for every continuation; first action counts toward K",
        "states_cache": str(args.states_cache),
        "states_cache_sha256": sha256(args.states_cache),
        "runner_sha256": sha256(args.runner),
        "scorer_sha256": sha256(Path(__file__)),
        "audit": {**audit, "ordered_snapshot_match_K4_K8": sum(a == b for a, b in zip([row["snapshot_hash"] for row in groups[4]], [row["snapshot_hash"] for row in groups[8]]))},
        "horizons": horizons,
    }
    with (args.out_dir / "TOKEN_BUDGET_MARKER_ALWAYS_ON_OFF_PER_STATE.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(per_state, key=lambda item: (item["K"], item["state_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = args.out_dir / "TOKEN_BUDGET_MARKER_ALWAYS_ON_OFF_SUMMARY.json"
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(args.out_dir.iterdir()) if path.is_file() and path.name != "SHA256SUMS"),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
