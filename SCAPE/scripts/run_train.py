#!/usr/bin/env python3
"""One-click Harness-1 / BC+ training.

Example:
  python scripts/run_train.py \\
    --harness Harness-1 --benchmark BC+ --model_name harness-1 \\
    --train_method scape+rl --component sentence_compress verify_tool
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.cli.launch import LaunchError, parse_train_args
from scape.training.rl_opd_types import TRAINING_MODE_RL


def main(argv: list[str] | None = None) -> int:
    try:
        args, spec = parse_train_args(argv)
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc

    args.lambda_opd = 0.0 if args.training_mode == TRAINING_MODE_RL else float(args.lambda_opd)
    args.train_steps = int(args.train_steps)
    args.max_steps = args.train_steps
    args.seeds = [int(args.seed)]
    args.on_policy_refresh = True
    args.gpu_schedule = "scheme_a"
    args.enforce_eager = True
    args.target_component = args.component

    spec.out.mkdir(parents=True, exist_ok=True)
    launch = {
        "harness": spec.harness,
        "benchmark": spec.benchmark,
        "model_name": spec.model_name,
        "train_method": spec.train_method,
        "training_mode": spec.training_mode,
        "opd_loss": args.opd_loss,
        "opd_states_per_trajectory": args.opd_states_per_trajectory,
        "component": spec.coalition,
        "component_ids": list(spec.components),
        "base_model": str(spec.base_model),
        "n_queries": args.n_queries,
        "score_split": "bcplus_test_166",
        "bcplus_split": "830 = 664 train + 166 test",
        "student_mask": "H_zero (all V8D OFF)" if spec.zero_components else "H_min (listed components OFF)",
        "teacher_mask": "H_zero (all V8D OFF)" if spec.zero_components else "H_full (listed components ON)",
        "out": str(spec.out),
    }
    (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2), flush=True)

    from scape.training.four_cell_runtime import run_from_rl_opd_args

    result = run_from_rl_opd_args(args)
    keep = (
        "ok",
        "component",
        "component_ids",
        "q1_joint_one_optim",
        "q2_on_policy_projection",
        "q3_teacher_does_not_change_reward",
        "n_train_queries",
        "n_eval_queries",
        "official_test_is_166",
        "using_full_train_split",
        "per_seed",
    )
    print(json.dumps({k: result[k] for k in keep if k in result}, indent=2), flush=True)
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
