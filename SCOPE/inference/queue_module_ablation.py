#!/usr/bin/env python3
"""Module-level paired ablation runner (replaces per-mechanism ablation)."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.harness_config import apply_harness_config, config_path, load_harness_config

MODULE_ABLATIONS = {
    "full": "modules_full.yaml",
    "minimal": "modules_minimal.yaml",
    "minus_evidence_state": "ablate_evidence_state.yaml",
    "minus_verification": "ablate_verification.yaml",
    "minus_context_budget": "ablate_context_budget.yaml",
}


@dataclass
class AblationJob:
    name: str
    config_name: str
    output_dir: Path
    env_extra: Dict[str, str] = field(default_factory=dict)


def bootstrap_ci(deltas: List[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(n_boot):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return (lo, hi)


def sample_query_ids(seed: int, limit: int) -> List[str]:
    from datagen.search_dataset import get_dataset

    dataset = get_dataset("browsecompplus")
    query_ids = dataset.get_test_query_ids()
    rng = random.Random(seed)
    return [str(q) for q in rng.sample(query_ids, min(limit, len(query_ids)))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue module-level ablations")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/module_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    query_ids = sample_query_ids(args.seed, args.limit)
    (output_root / "query_ids.json").write_text(
        json.dumps(query_ids, indent=2), encoding="utf-8"
    )

    jobs: List[AblationJob] = []
    for name, cfg_name in MODULE_ABLATIONS.items():
        cfg = load_harness_config(config_path(cfg_name))
        out = output_root / name
        out.mkdir(parents=True, exist_ok=True)
        cfg.save_resolved(out / "resolved_config.yaml")
        env_extra = apply_harness_config(cfg)
        jobs.append(AblationJob(name=name, config_name=cfg_name, output_dir=out, env_extra=env_extra))

    summary = {
        "conditions": list(MODULE_ABLATIONS.keys()),
        "query_count": len(query_ids),
        "seed": args.seed,
        "jobs": [
            {"name": j.name, "config": j.config_name, "output_dir": str(j.output_dir)}
            for j in jobs
        ],
    }

    if not args.dry_run:
        # Placeholder paired stats until evaluate_harness1 results are collected
        summary["paired_bootstrap_note"] = "Collect per-query metrics then call summarize"

    (output_root / "ablation_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
