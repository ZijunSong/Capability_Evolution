#!/usr/bin/env python3
"""Aggregate LOCAL_CAL64 LOO rollouts -> contribution table -> candidate selection."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from scape.probes.candidate_selector import select_candidates, write_placement_map
from scape.probes.contribution import contribution_report
from scape.common.sha256sums import write_sha256sums

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "local_cal64_loo"
PRE = REPO / "outputs" / "scape_prestage"


def _load_by_qid(job_dir: Path) -> dict[str, dict]:
    path = job_dir / "harness_rollouts.jsonl"
    if not path.exists():
        # try nested names used by SCOPE
        cands = list(job_dir.glob("**/harness_rollouts.jsonl"))
        if not cands:
            raise FileNotFoundError(path)
        path = cands[0]
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id") or row.get("qid") or row.get("id"))
            metrics = row.get("metrics") or row
            # normalize metric names
            item = {
                "curated_recall": float(
                    metrics.get("curated_recall")
                    or metrics.get("recall")
                    or metrics.get("evidence_recall")
                    or 0.0
                ),
                "trajectory_recall": float(
                    metrics.get("trajectory_recall")
                    or metrics.get("recall")
                    or metrics.get("curated_recall")
                    or 0.0
                ),
                "final_answer_recall": float(
                    metrics.get("final_answer_recall")
                    or metrics.get("exact_match")
                    or metrics.get("answer_correct")
                    or 0.0
                ),
                "harness_reward": float(
                    metrics.get("harness_reward")
                    or metrics.get("reward")
                    or metrics.get("score")
                    or 0.0
                ),
                "tool_calls": float(metrics.get("tool_calls") or metrics.get("n_tool_calls") or 0.0),
                "turns": float(metrics.get("turns") or metrics.get("n_turns") or 0.0),
                "context_tokens": float(
                    metrics.get("context_tokens") or metrics.get("prompt_tokens") or 0.0
                ),
                "latency_ms": float(metrics.get("latency_ms") or metrics.get("latency") or 0.0),
                "state_ops": float(metrics.get("state_ops") or 0.0),
            }
            out[qid] = item
    return out


def _mean_metric(by_qid: dict[str, dict], key: str) -> float:
    vals = [float(v[key]) for v in by_qid.values()]
    return float(statistics.mean(vals)) if vals else 0.0


def main() -> None:
    full_dir = OUT / "full"
    full = _load_by_qid(full_dir)
    reports = {}
    rows = []
    for job_dir in sorted(OUT.glob("minus_*")):
        cid = job_dir.name.replace("minus_", "", 1)
        minus = _load_by_qid(job_dir)
        # align ids
        shared = sorted(set(full) & set(minus))
        if len(shared) < 8:
            print(f"[warn] {cid}: only {len(shared)} shared qids")
        full_s = {k: full[k] for k in shared}
        minus_s = {k: minus[k] for k in shared}
        rep = contribution_report(cid, full_s, minus_s, n_boot=500, seed=42)
        reports[cid] = rep
        # influence placeholder until same-state probe finishes; use |Δ| as weak proxy * 0
        # Real influence filled later; use small epsilon so selector can still rank on contribution×semantic
        contrib = float(rep["metrics"].get("curated_recall", {}).get("mean_delta", 0.0))
        cost = abs(_mean_metric(full_s, "context_tokens") - _mean_metric(minus_s, "context_tokens")) + 1.0
        # provisional influence: disagreement proxy from reward delta magnitude
        infl = abs(
            float(rep["metrics"].get("harness_reward", {}).get("mean_delta", 0.0))
        ) + abs(contrib)
        rows.append(
            {
                "component_id": cid,
                "contribution": contrib,
                "influence_above_null": infl,
                "runtime_cost": cost,
                "quality_positive": bool(rep.get("quality_positive")),
                "n": len(shared),
                "provisional": True,
                "influence_is_proxy": True,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "COMPONENT_CONTRIBUTION.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "PROVISIONAL_ROWS.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    result = select_candidates(rows, top_k=2)
    paths = write_placement_map(result, PRE)
    # also copy selection into OUT
    (OUT / "CANDIDATE_SELECTION.json").write_text(
        json.dumps(result["candidates"], indent=2) + "\n", encoding="utf-8"
    )
    write_sha256sums(
        OUT,
        [
            OUT / "COMPONENT_CONTRIBUTION.json",
            OUT / "PROVISIONAL_ROWS.json",
            OUT / "CANDIDATE_SELECTION.json",
            PRE / "CAPABILITY_PLACEMENT_MAP.csv",
            PRE / "CANDIDATE_SELECTION.json",
        ],
    )
    print(json.dumps({"candidates": result["candidates"], "paths": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
