#!/usr/bin/env python3
"""Formal sentence_compress (or other registered component) 4-cell runner.

Trains Before / RL / PURE_SR_OPD / RL+OPD from the same theta_0 using:
  CISPO forward_backward -> sr_opd_ce forward_backward -> one optim_step

Then reloads adapters and evaluates the official 384-query pool.
"""

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
    run_four_cell,
    validate_wiring,
)
from scape.training.rl_opd_types import TRAINING_MODE_PURE_OPD, TRAINING_MODE_RL, TRAINING_MODE_RL_OPD


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formal SR-OPD + CISPO four-cell")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--component", default="sentence_compress", choices=sorted(TEACHER_REGISTRY))
    p.add_argument("--training-mode", default="four_cell", choices=("four_cell", TRAINING_MODE_RL, TRAINING_MODE_PURE_OPD, TRAINING_MODE_RL_OPD))
    p.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    p.add_argument("--sft-adapter", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-queries", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=None, help="Subset official eval; default is full 384.")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--lambda-opd", type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--opd-states-per-trajectory", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-manifest", type=Path, default=None)
    p.add_argument("--eval-manifest", type=Path, default=None)
    p.add_argument("--official-eval", action="store_true", default=True)
    p.add_argument("--no-official-eval", action="store_false", dest="official_eval")
    p.add_argument("--validate-only", action="store_true", help="Check collector/teacher/384 pool without loading a model.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rollout-backend", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--gpu-schedule", choices=("scheme_a", "resident_hf"), default="scheme_a")
    p.add_argument("--tensor-parallel-size", type=int, default=None, help="vLLM TP. Default: all visible GPUs.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--train-device-map", default="", help="HF device_map. Default auto under scheme_a.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager")
    p.add_argument("--no-on-policy-refresh", action="store_false", dest="on_policy_refresh")
    p.add_argument("--on-policy-refresh", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = coerce_runtime_args(parse_args())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    (args.out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.validate_only or args.dry_run:
        report = validate_wiring(args)
        (args.out / "VALIDATE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if report.get("teacher_leak_in_student_prefix"):
            raise SystemExit("Teacher compressed view leaked into Student prefix")
        return 0
    summary = run_four_cell(args)
    print(json.dumps({"done": True, "out": str(args.out), "q1": summary.get("q1_joint_one_optim"), "q2": summary.get("q2_on_policy_projection"), "q3": summary.get("q3_teacher_does_not_change_reward")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
