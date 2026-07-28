#!/usr/bin/env python3
"""Module-level evaluation entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.graph.registry import ModuleRegistry
from harness.harness_config import apply_harness_config, config_path, load_harness_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Harness with module config")
    parser.add_argument(
        "--module-config",
        default=str(config_path("modules_full.yaml")),
    )
    parser.add_argument("--dataset", default="browsecompplus")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/module_eval")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_harness_config(args.module_config)
    env = apply_harness_config(cfg)
    registry = ModuleRegistry.from_config(cfg)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_resolved(out / "resolved_config.yaml")

    summary = {
        "module_config_hash": cfg.config_hash(),
        "module_config_path": args.module_config,
        "dataset": args.dataset,
        "limit": args.limit,
        "seed": args.seed,
        "modules": {
            mid: {
                "enabled": m.config.enabled,
                "node_count": len(m.nodes),
                "lifecycle_managed": m.config.lifecycle_managed,
            }
            for mid, m in registry.modules.items()
        },
        "applied_env": env,
    }

    if args.dry_run:
        summary["dry_run"] = True
    else:
        # Delegate to existing evaluate_harness1 when credentials are available
        summary["note"] = (
            "Run inference/evaluate_harness1.py with applied env for full eval"
        )

    result_path = out / "module_eval_summary.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
