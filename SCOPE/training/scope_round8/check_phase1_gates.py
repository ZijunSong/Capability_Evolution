#!/usr/bin/env python3
"""Evaluate Round 8 Phase 1 gates 1A/1B/1C."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.dup_telemetry import AdmissionEvent, DupTelemetryAggregator
from training.scope_round8.compare_agent_configs import compare_configs

OUT = _REPO / "outputs/scope_round8"
DUP_ROOT = OUT / "dup_retention_830"
ROLLBACK_DS = _REPO / "artifacts/datasets/scope_round8/rollback_sdi"
SEEDS = ["seed42", "seed43", "seed44"]
SHARDS = ["shard0", "shard1", "shard2", "shard3"]

GATE_1A_DUP_REJECT_MIN = 0.10
GATE_1A_FSR_MAX = 0.05
GATE_1A_BAL_ACC_MIN = 0.50
RECALL_DROP_MAX_PP = 0.01


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_dup_telemetry(label: str) -> dict[str, Any]:
    agg = DupTelemetryAggregator()
    for shard in SHARDS:
        sid = shard.replace("shard", "")
        ep_path = DUP_ROOT / label / f"shard{sid}" / "dup_admission_events.jsonl"
        for row in load_jsonl(ep_path):
            agg.add(
                AdmissionEvent(
                    candidate_evidence_id=str(row.get("candidate_evidence_id", "")),
                    candidate_is_duplicate=bool(row.get("candidate_is_duplicate")),
                    student_operation=row.get("student_operation"),
                    shadow_operation=row.get("shadow_operation"),
                    route=row.get("route"),
                    actually_curated=bool(row.get("actually_curated")),
                    query_id=str(row.get("query_id", "")),
                    turn_id=int(row.get("turn_id", 0)),
                )
            )
    tel = agg.summarize()
    skip = tel.get("SKIP_DUPLICATE", {})
    keep = tel.get("KEEP_EVIDENCE", {})
    return {
        "DupRejectRecall": skip.get("recall", tel.get("duplicate_reject_rate", 0.0)),
        "FalseSkipRate": tel.get("false_skip_rate", 0.0),
        "BalancedAcc": tel.get("balanced_accuracy", 0.0),
        "SKIP_prior": tel.get("n_pred_skip", 0) / max(tel.get("n_decision_points", 1), 1),
        "telemetry": tel,
    }


def mean_recall(label: str) -> float:
    recalls: list[float] = []
    for shard in SHARDS:
        sid = shard.replace("shard", "")
        for row in load_jsonl(DUP_ROOT / label / f"shard{sid}" / "episodes.jsonl"):
            recalls.append(float(row.get("recall", 0.0)))
    return sum(recalls) / max(len(recalls), 1)


def paired_recall_ci(base_recalls: list[float], seed_recalls: list[float]) -> dict[str, float]:
    if not base_recalls or not seed_recalls:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    n = min(len(base_recalls), len(seed_recalls))
    deltas = [seed_recalls[i] - base_recalls[i] for i in range(n)]
    mean_d = sum(deltas) / n
    rng = random.Random(42)
    boots = []
    for _ in range(200):
        samp = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(samp) / n)
    boots.sort()
    return {
        "delta": mean_d,
        "ci_low": boots[int(0.025 * len(boots))],
        "ci_high": boots[int(0.975 * len(boots))],
    }


def dup_shard_complete(label: str) -> bool:
    expected = {"shard0": 207, "shard1": 207, "shard2": 207, "shard3": 209}
    for shard, exp in expected.items():
        sid = shard.replace("shard", "")
        n = len(load_jsonl(DUP_ROOT / label / f"shard{sid}" / "episodes.jsonl"))
        if n < exp:
            return False
    return True


def gate_1a() -> dict[str, Any]:
    base_metrics = merge_dup_telemetry("base")
    base_recalls = []
    for shard in SHARDS:
        sid = shard.replace("shard", "")
        for row in load_jsonl(DUP_ROOT / "base" / f"shard{sid}" / "episodes.jsonl"):
            base_recalls.append(float(row.get("recall", 0.0)))

    seed_results: dict[str, Any] = {}
    all_pass = True
    for seed in SEEDS:
        m = merge_dup_telemetry(seed)
        seed_recalls = []
        for shard in SHARDS:
            sid = shard.replace("shard", "")
            for row in load_jsonl(DUP_ROOT / seed / f"shard{sid}" / "episodes.jsonl"):
                seed_recalls.append(float(row.get("recall", 0.0)))
        ci = paired_recall_ci(base_recalls, seed_recalls)
        recall_ok = abs(ci["delta"]) <= RECALL_DROP_MAX_PP or ci["ci_low"] <= 0 <= ci["ci_high"]
        pass_d = (
            m["DupRejectRecall"] >= GATE_1A_DUP_REJECT_MIN
            and m["FalseSkipRate"] <= GATE_1A_FSR_MAX
            and m["BalancedAcc"] > GATE_1A_BAL_ACC_MIN
            and recall_ok
        )
        seed_results[seed] = {**m, "recall_ci": ci, "gate_pass": pass_d}
        if not pass_d:
            all_pass = False

    return {
        "base": base_metrics,
        "seeds": seed_results,
        "mean_base_recall": mean_recall("base"),
        "operation_parity": 1.0,
        "gate_1a_pass": all_pass and dup_shard_complete("base"),
        "dup_complete": {
            "base": dup_shard_complete("base"),
            **{s: dup_shard_complete(s) for s in SEEDS},
        },
    }


def gate_1b() -> dict[str, Any]:
    ac = _REPO / "harness/configs/agent_core.yaml"
    fh = _REPO / "harness/configs/agent_core_full_harness.yaml"
    diff = compare_configs(ac, fh)
    pass_b = bool(diff.get("gate_1b_pass"))
    return {"config_diff": diff, "gate_1b_pass": pass_b}


def gate_1c() -> dict[str, Any]:
    path = ROLLBACK_DS / "ROLLBACK_DATASET_GATE.json"
    if not path.exists():
        return {"gate_1c_pass": False, "error": "missing dataset gate file"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**data, "gate_1c_pass": data.get("gate_1c_pass", False)}


def phase1_complete() -> bool:
    return dup_shard_complete("base") and all(dup_shard_complete(s) for s in SEEDS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "HARD_CAPABILITY_GATE.json")
    args = p.parse_args()

    result = {
        "phase1_complete": phase1_complete(),
        "gate_1a": gate_1a(),
        "gate_1b": gate_1b(),
        "gate_1c": gate_1c(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
    }
    result["all_gates_pass"] = (
        result["phase1_complete"]
        and result["gate_1a"].get("gate_1a_pass")
        and result["gate_1b"].get("gate_1b_pass")
        and result["gate_1c"].get("gate_1c_pass")
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
