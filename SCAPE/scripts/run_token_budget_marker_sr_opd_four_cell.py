#!/usr/bin/env python3
"""Formal token_budget_marker SR-OPD + CISPO four-cell runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.training.four_cell_runtime import (
    TEACHER_REGISTRY,
    build_manifest,
    coerce_runtime_args,
    run_seeded_four_cell,
    validate_wiring,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="token_budget_marker SR-OPD + CISPO four-cell")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--component", default="token_budget_marker", choices=["token_budget_marker"])
    p.add_argument("--training-mode", default="four_cell", help="four_cell | rl | rl_opd | pure_opd_only | rl_opd_only")
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", default="")
    p.add_argument("--gpu", default="0", help="GPU index or 'auto' for multi-GPU device mapping")
    p.add_argument("--n-queries", type=int, default=664)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--lambda-opd", type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--opd-states-per-trajectory", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--query-manifest", type=Path, default=None)
    p.add_argument("--eval-manifest", type=Path, default=None)
    p.add_argument("--train-states", type=Path, default=None)
    p.add_argument("--n-train-states", type=int, default=None)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rollout-backend", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--gpu-schedule", choices=("scheme_a", "resident_hf"), default="scheme_a")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--train-device-map", default="")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager")
    p.add_argument("--vllm-python", default="", help="Python executable for the vLLM rollout worker")
    p.add_argument("--no-on-policy-refresh", action="store_false", dest="on_policy_refresh")
    p.add_argument("--on-policy-refresh", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    args = coerce_runtime_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, extra={"runner": "run_token_budget_marker_sr_opd_four_cell.py"})
    (args.out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.validate_only or args.dry_run:
        report = validate_wiring(args)
        (args.out / "VALIDATE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    summary = run_seeded_four_cell(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
