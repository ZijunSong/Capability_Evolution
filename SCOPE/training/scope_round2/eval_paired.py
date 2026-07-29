#!/usr/bin/env python3
"""Paired comparison of Base vs Round1 (or any two variants) on fixed 100q."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.distillability.metrics import aggregate_episodes, episodes_by_query


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(episodes: list[dict[str, Any]]) -> dict[str, float]:
    agg = aggregate_episodes(episodes)
    n = max(len(episodes), 1)
    dup_curate = sum(float(e.get("dup_curate_rate", 0)) for e in episodes) / n
    repeated = sum(float(e.get("repeated_evidence_rate", 0)) for e in episodes) / n
    unique_ratio = sum(float(e.get("unique_curated_ratio", 0)) for e in episodes) / n
    stop_rate = sum(1 for e in episodes if int(e.get("turns", 0)) < 8) / n
    premature = sum(
        1 for e in episodes if int(e.get("turns", 0)) < 8 and float(e.get("recall", 0)) == 0
    ) / n
    return {
        **agg,
        "duplicate_curate_rate": dup_curate,
        "repeated_evidence_rate": repeated,
        "unique_evidence_ratio": unique_ratio,
        "mean_turns": agg.get("turns", 0),
        "mean_n_curated": agg.get("n_curated", 0),
        "mean_n_pool": agg.get("n_pool", 0),
        "stop_rate": stop_rate,
        "premature_stop_rate": premature,
        "verify_calls": sum(
            float((e.get("metrics") or {}).get("verify_calls", 0)) for e in episodes
        )
        / n,
    }


def bootstrap_ci(deltas: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randint(0, len(deltas) - 1)] for _ in range(len(deltas))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return lo, hi


def paired_analysis(
    base: dict[str, dict], other: dict[str, dict], metric: str
) -> dict[str, Any]:
    deltas = []
    wins = losses = ties = 0
    for qid in sorted(set(base) & set(other)):
        b = float(base[qid].get(metric, 0))
        o = float(other[qid].get(metric, 0))
        d = o - b
        deltas.append(d)
        if d > 1e-6:
            wins += 1
        elif d < -1e-6:
            losses += 1
        else:
            ties += 1
    lo, hi = bootstrap_ci(deltas)
    return {
        "metric": metric,
        "mean_delta": sum(deltas) / max(len(deltas), 1),
        "bootstrap_ci_95": [lo, hi],
        "win": wins,
        "loss": losses,
        "tie": ties,
        "n": len(deltas),
    }


def render_md(report: dict[str, Any]) -> str:
    b = report["base"]
    r = report["round1"]
    lines = [
        "# Base vs Old Round1 — H_min_v2 100q\n",
        "| Metric | Base | Old Round1 | Delta |",
        "|--------|------|------------|-------|",
    ]
    for key in [
        "recall",
        "reward",
        "trajectory_recall",
        "final_answer_recall",
        "mean_turns",
        "mean_n_curated",
        "mean_n_pool",
        "duplicate_curate_rate",
        "unique_evidence_ratio",
        "repeated_evidence_rate",
        "premature_stop_rate",
    ]:
        bv = b.get(key, 0)
        rv = r.get(key, 0)
        lines.append(f"| {key} | {bv:.4f} | {rv:.4f} | {rv - bv:+.4f} |")
    lines.append("\n## Paired deltas (Round1 - Base)\n")
    for pa in report.get("paired", []):
        lines.append(
            f"- **{pa['metric']}**: mean={pa['mean_delta']:+.4f}, "
            f"CI95=[{pa['bootstrap_ci_95'][0]:+.4f}, {pa['bootstrap_ci_95'][1]:+.4f}], "
            f"W/L/T={pa['win']}/{pa['loss']}/{pa['tie']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--round1", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()

    base_eps = load_jsonl(args.base / "episodes.jsonl")
    r1_eps = load_jsonl(args.round1 / "episodes.jsonl")
    base_by_q = episodes_by_query(base_eps)
    r1_by_q = episodes_by_query(r1_eps)

    report = {
        "base": summarize(base_eps),
        "round1": summarize(r1_eps),
        "paired": [
            paired_analysis(base_by_q, r1_by_q, m)
            for m in [
                "recall",
                "reward",
                "n_curated",
                "dup_curate_rate",
                "unique_curated_ratio",
            ]
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_md(report), encoding="utf-8")
    print(json.dumps(report["base"], indent=2))


if __name__ == "__main__":
    main()
