#!/usr/bin/env python3
"""One-click Harness-1 / Harness-G training.

Runs only the training cell for ``--train_method`` (rl / opd / rl+opd /
scape+rl / trim). Does not run the four-cell protocol (no Before baseline, no
closed-loop eval). Score with ``scripts/run_eval.py``.

Single-node Scheme A (one process, all visible GPUs): this file, default
``--training-backend hf_debug``.
Single-node 8-GPU FSDP2: ``--training-backend verl`` (this file auto-torchruns).
Multi-node Scheme A: ``scripts/run_train_multinode.py``.

Example:
  python scripts/run_train.py \\
    --harness Harness-1 --benchmark BC+ --model_name /path/to/checkpoint \\
    --train_method trim --component all

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_train.py \\
    --training-backend verl --train_method rl --train-steps 100 \\
    --harness Harness-1 --benchmark bcplus_full --component all \\
    --max-turns 40 --model_name /path/to/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.offline_credentials import ensure_local_offline_credentials

ensure_local_offline_credentials()

from trim.cli.launch import LaunchError, parse_train_args, student_mask_for_ids, teacher_mask_for_ids
from trim.eval.official_query_pool import SCORE_SPLIT_166, SCORE_SPLIT_830
from trim.eval.sec_corpus import (
    SEC_TRAIN_POOL_NAME,
    default_sec_corpus_root,
    default_sec_rl_data,
)
from trim.training.dist_runtime import (
    DIST_BACKEND_ENV,
    DistLaunchConfig,
    needs_torchrun,
    parse_dist_argv,
    pin_local_cuda_device,
    torchrun_cmd,
    training_backend_from_argv,
    under_torchrun,
    visible_cuda_count,
)
from trim.training.rl_opd_types import TRAINING_MODE_RL

VERL_BACKENDS = {"verl", "fsdp2", "verl_fsdp2"}


def apply_train_launcher_defaults(args) -> None:
    args.lambda_opd = 0.0 if args.training_mode == TRAINING_MODE_RL else float(args.lambda_opd)
    args.train_steps = int(args.train_steps)
    args.max_steps = args.train_steps
    args.seeds = [int(args.seed)]
    if getattr(args, "on_policy_refresh", None) is None:
        args.on_policy_refresh = True
    backend = str(getattr(args, "training_backend", "hf_debug") or "hf_debug").lower().replace("-", "_")
    if backend in VERL_BACKENDS:
        args.gpu_schedule = "verl_fsdp2"
        args.enforce_eager = bool(getattr(args, "enforce_eager", False))
        args.tensor_parallel_size = int(getattr(args, "tensor_parallel_size", None) or 1)
        args.rollout_replicas = 1
    else:
        args.gpu_schedule = "scheme_a"
        args.enforce_eager = True
    args.target_component = args.component
    args.train_only = True
    args.official_eval = False
    if not hasattr(args, "rollout_replicas") or getattr(args, "rollout_replicas", None) in {None, 0}:
        args.rollout_replicas = 1


def _forward_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pythonpath = env.get("PYTHONPATH", "")
    parts = [str(_TRIM)]
    scape = _TRIM.parent / "SCAPE-EasyOPD"
    if scape.is_dir():
        parts.append(str(scape))
    if pythonpath:
        parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _exec_fsdp2_torchrun(cfg: DistLaunchConfig, train_argv: list[str]) -> int:
    cmd = torchrun_cmd(script=str(Path(__file__).resolve()), train_argv=train_argv, cfg=cfg, node_rank=0)
    env = _forward_env()
    env[DIST_BACKEND_ENV] = "nccl"
    env["TRIM_GPU_KEEPALIVE"] = "0"
    env["MASTER_ADDR"] = str(cfg.master_addr)
    env["MASTER_PORT"] = str(int(cfg.master_port))
    print("[verl-fsdp2-launch] " + " ".join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 1


def build_train_launch_record(args, spec) -> dict:
    from trim.training.dist_runtime import dist_info

    info = dist_info()
    return {
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
        "train_steps": int(args.train_steps),
        "max_turns": int(args.max_turns),
        "group_size": int(args.group_size),
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
        "student_mask_label": (
            "H_zero (all advanced components OFF)"
            if spec.zero_components
            else "H_min (listed advanced components OFF)"
        ),
        "teacher_mask_label": (
            "H_zero (all advanced components OFF)"
            if spec.zero_components
            else "H_full (listed advanced components ON)"
        ),
        "student_mask": student_mask_for_ids(spec.components, harness=spec.harness),
        "teacher_mask": teacher_mask_for_ids(
            spec.components, harness=spec.harness, preset=spec.component_preset
        ),
        "on_policy_refresh": bool(args.on_policy_refresh),
        "train_env": getattr(args, "train_env", "upstream"),
        "teacher_kind": getattr(args, "teacher_kind", "upstream"),
        "out": str(spec.out),
        "train_only": True,
        "official_eval": False,
        "max_new_tokens": int(args.max_new_tokens),
        "max_model_len": int(getattr(args, "max_model_len", 8192) or 8192),
        "max_num_seqs": int(getattr(args, "max_num_seqs", 256) or 256),
        "vllm_generate_timeout_s": float(getattr(args, "vllm_generate_timeout_s", 3600.0) or 3600.0),
        "vllm_disable_custom_all_reduce": getattr(args, "vllm_disable_custom_all_reduce", None),
        "rollout_backend": str(getattr(args, "rollout_backend", "vllm") or "vllm"),
        "training_backend": str(getattr(args, "training_backend", "hf_debug") or "hf_debug"),
        "gpu_schedule": str(getattr(args, "gpu_schedule", "scheme_a") or "scheme_a"),
        "enforce_eager": bool(getattr(args, "enforce_eager", False)),
        "tensor_parallel_size": getattr(args, "tensor_parallel_size", None),
        "rollout_replicas": int(getattr(args, "rollout_replicas", 1) or 1),
        "dist": {
            "rank": int(info.rank),
            "world_size": int(info.world_size),
            "local_rank": int(info.local_rank),
            "local_world_size": int(info.local_world_size),
            "node_rank": int(info.node_rank),
            "nnodes": int(getattr(args, "dist_nnodes", info.world_size) or info.world_size),
            "nproc_per_node": int(getattr(args, "dist_nproc_per_node", info.local_world_size) or info.local_world_size),
        },
    }


def run_parsed_train(args, spec) -> int:
    apply_train_launcher_defaults(args)
    from trim.training.dist_runtime import is_coordinator

    if is_coordinator():
        spec.out.mkdir(parents=True, exist_ok=True)
        launch = build_train_launch_record(args, spec)
        (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(launch, indent=2), flush=True)
    else:
        launch = build_train_launch_record(args, spec)
        print(json.dumps({"rank_skip_launch_write": True, **launch.get("dist", {})}, indent=2), flush=True)

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


def main(argv: list[str] | None = None) -> int:
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive

    raw = list(sys.argv[1:] if argv is None else argv)
    cfg, train_argv = parse_dist_argv(raw)
    backend = training_backend_from_argv(train_argv)
    if backend in VERL_BACKENDS:
        os.environ["TRIM_GPU_KEEPALIVE"] = "0"
        os.environ.setdefault(DIST_BACKEND_ENV, "nccl")
        if not cfg.already_worker and not under_torchrun():
            nproc = int(cfg.nproc_per_node)
            if nproc <= 1:
                nproc = visible_cuda_count()
                if nproc <= 0:
                    try:
                        import torch

                        nproc = int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
                    except Exception:
                        nproc = 1
            cfg = replace(
                cfg,
                nproc_per_node=max(1, int(nproc)),
                dist_backend="nccl",
                rollout_replicas=1,
            )
            if needs_torchrun(cfg):
                return _exec_fsdp2_torchrun(cfg, train_argv)
        pin_local_cuda_device()
    acquire_keepalive()
    try:
        return _main(train_argv)
    finally:
        release_keepalive()


def _main(argv: list[str] | None = None) -> int:
    try:
        args, spec = parse_train_args(argv)
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc
    return run_parsed_train(args, spec)


if __name__ == "__main__":
    raise SystemExit(main())
