#!/usr/bin/env python3
"""Canonical RL + Student-Realizable Projected OPD launcher.

Joint contract: CISPO forward_backward, SR-OPD CE forward_backward,
then exactly one optim_step. Does not import the legacy SCOPE OPDTrainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.training.rl_opd_types import (
    PROTOCOL_COMPLETE_RL_OPD,
    TRAINING_MODE_PURE_OPD,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True SCAPE RL + SR-OPD")
    p.add_argument(
        "--training-mode",
        choices=(TRAINING_MODE_RL, TRAINING_MODE_PURE_OPD, TRAINING_MODE_RL_OPD),
        default=TRAINING_MODE_RL_OPD,
    )
    p.add_argument("--rl-loss-fn", default="cispo")
    p.add_argument("--lambda-opd", type=float, default=0.1)
    p.add_argument("--target-component", default="sentence_compress")
    p.add_argument("--component", default="", help="Alias of --target-component")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--sft-adapter", default="")
    p.add_argument("--query-manifest", type=Path, default=None)
    p.add_argument("--eval-manifest", type=Path, default=None)
    p.add_argument("--n-queries", type=int, default=664)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--train-states", type=Path, default=None)
    p.add_argument("--seeds", default="42", help="Comma-separated seeds; sentence_compress formal uses 42,43")
    p.add_argument("--student-harness-config", default="H_min")
    p.add_argument("--teacher-harness-config", default="H_full")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--num-substeps", type=int, default=4)
    p.add_argument("--opd-states-per-trajectory", type=int, default=3)
    p.add_argument("--projection-max-events", type=int, default=8)
    p.add_argument("--projection-max-macro-actions", type=int, default=3)
    p.add_argument("--teacher-shadow-max-turns", type=int, default=6)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--kl-penalty-coef", type=float, default=0.005)
    p.add_argument("--kl-reference-checkpoint", default="")
    p.add_argument("--base-checkpoint", default="")
    p.add_argument("--base-model", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Override defaults to batch 4 / group 2 / 2 steps for debug.",
    )
    p.add_argument("--rollout-backend", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--gpu-schedule", choices=("scheme_a", "resident_hf"), default="scheme_a")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--train-device-map", default="")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--no-on-policy-refresh", action="store_false", dest="on_policy_refresh")
    p.add_argument("--on-policy-refresh", action="store_true", default=True)
    return p.parse_args()


def apply_scale(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.batch_size = 4
        args.group_size = 2
        args.num_substeps = 1
        args.max_steps = 2
        args.max_turns = 2
        args.teacher_shadow_max_turns = 2
        args.opd_states_per_trajectory = 2
    return args


def resolved_lambda(args: argparse.Namespace) -> float:
    if args.training_mode == TRAINING_MODE_RL:
        return 0.0
    if args.training_mode == TRAINING_MODE_PURE_OPD:
        return float(args.lambda_opd)
    return float(args.lambda_opd)


def write_manifest(args: argparse.Namespace, path: Path) -> dict:
    lam = resolved_lambda(args)
    manifest = {
        "training_mode": args.training_mode,
        "rl_loss_fn": args.rl_loss_fn,
        "lambda_opd": lam,
        "base_checkpoint": args.base_checkpoint,
        "policy_version_start": "v0",
        "kl_reference_checkpoint": args.kl_reference_checkpoint or None,
        "kl_penalty_coef": args.kl_penalty_coef,
        "student_harness": args.student_harness_config,
        "teacher_harness": args.teacher_harness_config,
        "target_component": args.target_component,
        "projection_schema_version": "scape_projection_v1",
        "opd_loss": "sr_opd_ce",
        "opd_state_source": "current_on_policy_rl_rollout",
        "group_size": args.group_size,
        "num_substeps": args.num_substeps,
        "opd_states_per_trajectory": args.opd_states_per_trajectory,
        "projection_max_events": args.projection_max_events,
        "projection_max_macro_actions": args.projection_max_macro_actions,
        "teacher_shadow_max_turns": args.teacher_shadow_max_turns,
        "max_turns": args.max_turns,
        "scale": "smoke" if args.smoke else "full",
        "n_queries": args.n_queries,
        "score_split": "bcplus_test_166",
        "bcplus_split": "830 = 664 train + 166 test",
        "joint_update_contract": "rl_fb+opd_fb+single_optim",
        "legacy_tool_token_kl_hook_used": False,
        "protocol_complete_rl_opd": args.training_mode == TRAINING_MODE_RL_OPD and lam > 0,
        "protocol_name": PROTOCOL_COMPLETE_RL_OPD,
        "seed": args.seed,
        "max_steps": args.max_steps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = apply_scale(parse_args())
    if args.rl_loss_fn != "cispo":
        print(
            f"[rl_opd] rl_loss_fn={args.rl_loss_fn} is an explicit local override; "
            "canonical default remains cispo",
            flush=True,
        )
    out = Path(args.out)
    manifest = write_manifest(args, out / "RUN_MANIFEST.json")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.component:
        args.target_component = args.component
    else:
        args.component = args.target_component
    args.train_steps = args.max_steps
    if not args.base_model:
        args.base_model = args.base_checkpoint
    args.seeds = [int(x) for x in str(getattr(args, "seeds", "42")).split(",") if str(x).strip()]

    print(
        "[rl_opd] launching scape.training.four_cell_runtime "
        f"(component={args.component} mode={args.training_mode} "
        f"{'validate-only' if args.dry_run or args.validate_only else 'live'})",
        flush=True,
    )
    from scape.training.four_cell_runtime import run_from_rl_opd_args

    if args.dry_run or args.validate_only or args.training_mode == TRAINING_MODE_PURE_OPD or resolved_lambda(args) > 0:
        result = run_from_rl_opd_args(args)
        print(json.dumps({"live": not (args.dry_run or args.validate_only), **{k: result.get(k) for k in ("ok", "q1_joint_one_optim", "q2_on_policy_projection", "q3_teacher_does_not_change_reward") if k in result}}, indent=2), flush=True)
        return

    print(
        "[rl_opd] lambda_opd=0: skip Teacher/projector/OPD FB; "
        "RL-only path can use this HF loop or SCOPE/training/train_rl.py CISPO",
        flush=True,
    )
    result = run_from_rl_opd_args(args)
    print(json.dumps({"live": True, "training_mode": args.training_mode}, indent=2), flush=True)
    del result


if __name__ == "__main__":
    main()
