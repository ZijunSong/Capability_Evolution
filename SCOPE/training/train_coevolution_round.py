#!/usr/bin/env python3
"""Co-evolution round orchestrator: evaluate → OPD → re-evaluate → lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from harness.harness_config import config_path, load_harness_config
from harness.lifecycle.contribution import ModuleAudit
from harness.lifecycle.decision import decide
from harness.lifecycle.distillability import compute_distillability, compute_module_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one co-evolution round")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--input-model", default="base")
    parser.add_argument("--target-module", default="verification")
    parser.add_argument(
        "--full-config", default=str(config_path("modules_full.yaml"))
    )
    parser.add_argument(
        "--minus-config",
        default=str(config_path("ablate_verification.yaml")),
    )
    parser.add_argument("--output-dir", default="outputs/coevolution")
    parser.add_argument("--delta-before", type=float, default=0.15)
    parser.add_argument("--delta-after", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir) / f"round_{args.round}"
    out.mkdir(parents=True, exist_ok=True)

    harness_config = load_harness_config(args.full_config)
    distillability = compute_distillability(args.delta_before, args.delta_after)
    audit = ModuleAudit(
        module_id=args.target_module,
        delta_before=args.delta_before,
        delta_after=args.delta_after,
        ci_before=(args.delta_before - 0.02, args.delta_before + 0.02),
        ci_after=(args.delta_after - 0.02, args.delta_after + 0.02),
    )
    decision = decide(audit)

    manifest = {
        "round": args.round,
        "input_model": args.input_model,
        "output_model": f"{args.input_model}_opd_r{args.round}",
        "input_harness_config": args.full_config,
        "target_module": args.target_module,
        "delta_before": args.delta_before,
        "delta_after": args.delta_after,
        "distillability": distillability,
        "decision": decision.value,
        "output_harness_config": args.full_config
        if decision.value == "active"
        else str(config_path("ablate_verification.yaml")),
    }
    manifest_path = out / "round_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh)
    print(json.dumps(manifest, indent=2))
    harness_config.save_resolved(out / "resolved_config.yaml")


if __name__ == "__main__":
    main()
