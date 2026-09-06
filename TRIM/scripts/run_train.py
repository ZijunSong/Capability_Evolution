#!/usr/bin/env python3
"""One-click Harness-1 / Harness-G training.

Runs only the training cell for ``--train_method`` (rl / opd / rl+opd /
scape+rl / trim). Does not run the four-cell protocol (no Before baseline, no
closed-loop eval). Score with ``scripts/run_eval.py``.

Example:
  python scripts/run_train.py \\
    --harness Harness-1 --benchmark BC+ --model_name /path/to/checkpoint \\
    --train_method trim --component all

  python scripts/run_train.py \\
    --harness Harness-G --benchmark BC+ --model_name /path/to/checkpoint \\
    --train_method trim --component all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.cli.launch import LaunchError, parse_train_args
from trim.eval.official_query_pool import SCORE_SPLIT_166, SCORE_SPLIT_830
from trim.eval.sec_corpus import (
    SEC_TRAIN_POOL_NAME,
    default_sec_corpus_root,
    default_sec_rl_data,
)
from trim.training.rl_opd_types import TRAINING_MODE_RL


def main(argv: list[str] | None = None) -> int:
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive

    acquire_keepalive()
    try:
        return _main(argv)
    finally:
        release_keepalive()


def _main(argv: list[str] | None = None) -> int:
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
    args.train_only = True
    args.official_eval = False

    spec.out.mkdir(parents=True, exist_ok=True)
    launch = {
        "harness": spec.harness,
        "benchmark": spec.benchmark,
        "model_name": spec.model_name,
        "train_method": spec.train_method,
        "training_mode": spec.training_mode,
        "train_data": args.train_data,
        "opd_loss": args.opd_loss,
        "opd_states_per_trajectory": args.opd_states_per_trajectory,
        "lambda_opd": args.lambda_opd,
        "opd_gate_beta": float(getattr(args, "opd_gate_beta", 5.0) or 5.0),
        "component": spec.coalition,
        "component_ids": list(spec.components),
        "base_model": str(spec.base_model),
        "n_queries": args.n_queries,
        "train_groups_per_step": int(getattr(args, "train_groups_per_step", 32) or 0),
        "train_micro_batch_size": int(getattr(args, "train_micro_batch_size", 4) or 4),
        "train_heartbeat_every": int(getattr(args, "train_heartbeat_every", 8) or 8),
        "score_split": str(getattr(args, "score_split", None) or SCORE_SPLIT_166),
        "bcplus_split": (
            "830 = 664+166"
            if args.train_data == "sec"
            else "830 = 664 train + 166 test"
        ),
        "train_pool": SEC_TRAIN_POOL_NAME if args.train_data == "sec" else "bcplus_train_664",
        "rl_data": str(getattr(args, "rl_data", None) or default_sec_rl_data())
        if args.train_data == "sec"
        else None,
        "sec_corpus_root": str(getattr(args, "sec_corpus_root", None) or default_sec_corpus_root())
        if args.train_data == "sec"
        else None,
        "student_mask": (
            "H_zero (all advanced components OFF)"
            if spec.zero_components
            else "H_min (listed advanced components OFF)"
        ),
        "teacher_mask": (
            "H_zero (all advanced components OFF)"
            if spec.zero_components
            else "H_full (listed advanced components ON)"
        ),
        "out": str(spec.out),
        "train_only": True,
        "official_eval": False,
    }
    (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2), flush=True)

    from trim.training.four_cell_runtime import run_from_rl_opd_args

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
        "eval_is_bcplus_830",
        "train_pool",
        "score_split",
        "using_full_train_split",
        "per_seed",
        "train_only",
        "official_eval",
    )
    print(json.dumps({k: result[k] for k in keep if k in result}, indent=2), flush=True)
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
