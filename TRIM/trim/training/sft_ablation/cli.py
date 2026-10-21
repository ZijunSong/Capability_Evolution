"""Three independent SFT parsers. They do not call ``parse_train_args``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from trim.training.sft_ablation.contracts import PROFILES
from trim.training.sft_ablation.geometry import flag_present

FORBIDDEN = (
    "--train_method",
    "--train-method",
    "--lambda-opd",
    "--lambda_opd",
    "--opd-gate-beta",
    "--opd_gate_beta",
    "--rl-reward",
    "--rl_reward",
)

TEACHER_ONLY = (
    "--teacher-model",
    "--teacher_model",
    "--teacher-adapter",
    "--teacher_adapter",
    "--teacher-traces",
    "--teacher_traces",
    "--sft-data",
    "--sft_data",
)


def _reject_flags(argv: Sequence[str], names: Sequence[str], *, experiment: str) -> None:
    banned = set(names)
    for token in argv:
        head = str(token).split("=", 1)[0]
        if head in banned:
            raise SystemExit(f"{experiment} does not accept {head}")


def _add_shared(parser: argparse.ArgumentParser, *, experiment: str) -> None:
    parser.add_argument("--model_name", "--model-name", dest="model_name", required=True)
    parser.add_argument("--init-adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-data", default="sec")
    parser.add_argument("--rl-data", default="")
    parser.add_argument("--query-manifest", default="")
    parser.add_argument("--eval-manifest", default="")
    parser.add_argument("--sec-corpus-root", default="")
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="constant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--teacher-component", choices=("all",), default="all")
    parser.add_argument("--student-component", choices=("zero",), default="zero")
    parser.add_argument("--train-env", choices=("upstream", "local_legacy"), default="upstream")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--retrieval-backend", default="")
    parser.add_argument("--bm25-index", default="")
    parser.add_argument("--docstore", default="")
    parser.add_argument("--corpus", default="")
    parser.add_argument("--verifier", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--experiment", default=experiment)
    parser.add_argument("--ce-backend", choices=("legacy", "accelerate_ddp"), default="legacy")
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--source-contract", default="")
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--gradient-checkpointing", choices=("off", "non_reentrant"), default="off")
    parser.add_argument("--logits-mode", choices=("action_positions", "full"), default="action_positions")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--model-precision", choices=("bf16",), default="bf16")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every-microbatches", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--resume-round", default="")
    parser.add_argument("--online-layout", choices=("split_pools", "phase_jobs"), default="phase_jobs")
    parser.add_argument("--rollout-endpoints", default="")
    parser.add_argument("--queries-per-round", type=int, default=0)
    parser.set_defaults(experiment=experiment)


def _parser(experiment: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=experiment, allow_abbrev=False)
    _add_shared(parser, experiment=experiment)
    if experiment == "student_onpolicy_sft":
        parser.add_argument("--stage", choices=("all",), default="all")
        parser.add_argument("--states-per-trajectory", type=int, default=-1)
        parser.add_argument("--max-empty-rounds", type=int, default=3)
    else:
        parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
        parser.add_argument("--teacher-model", default="")
        parser.add_argument("--teacher-adapter", default="")
        parser.add_argument("--teacher-traces", default="")
        parser.add_argument("--sft-data", default="")
        parser.add_argument("--max-anchor-scan-events", type=int, default=8)
    return parser


def parse_sft_args(experiment: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    if experiment not in PROFILES:
        raise SystemExit(f"unknown SFT experiment: {experiment}")
    raw = list(sys.argv[1:] if argv is None else argv)
    _reject_flags(raw, FORBIDDEN, experiment=experiment)
    if experiment == "student_onpolicy_sft":
        _reject_flags(raw, TEACHER_ONLY, experiment=experiment)
    args = _parser(experiment).parse_args(raw)
    if str(args.experiment) != experiment:
        raise SystemExit(f"{experiment} refuses experiment override {args.experiment}")
    if int(args.num_updates) < 1 or int(args.batch_size) < 1 or int(args.micro_batch_size) < 1:
        raise SystemExit("num-updates, batch-size, and micro-batch-size must be positive")
    if int(args.group_size) < 1:
        raise SystemExit("group-size must be positive")
    if int(args.tensor_parallel_size) < 1:
        raise SystemExit("tensor-parallel-size must be positive")
    if args.retrieval_backend == "local_bm25":
        missing = [
            name
            for name, value in (
                ("--bm25-index", args.bm25_index),
                ("--docstore", args.docstore),
                ("--corpus", args.corpus),
                ("--verifier", args.verifier),
            )
            if not str(value)
        ]
        if missing:
            raise SystemExit("local_bm25 requires " + ", ".join(missing))
    if experiment != "student_onpolicy_sft" and int(args.max_anchor_scan_events) < 1:
        raise SystemExit("max-anchor-scan-events must be positive")
    if args.train_env == "local_legacy":
        args.train_env_report = "local_legacy_separate_report"
    else:
        args.train_env_report = "upstream"
    explicit_batch = flag_present(raw, "--batch-size", "--batch_size")
    explicit_global = flag_present(raw, "--global-batch-size", "--global_batch_size")
    if explicit_batch and explicit_global and int(args.batch_size) != int(args.global_batch_size):
        raise SystemExit("--batch-size and --global-batch-size disagree")
    if explicit_global and not explicit_batch:
        args.batch_size = int(args.global_batch_size)
    elif args.global_batch_size is None:
        args.global_batch_size = int(args.batch_size)
    if flag_present(raw, "--queries-per-round"):
        if int(args.queries_per_round) < 1:
            raise SystemExit("queries-per-round must be positive")
    else:
        args.queries_per_round = int(args.batch_size)
    if args.ce_backend == "accelerate_ddp":
        if int(args.micro_batch_size) != 1:
            raise SystemExit("accelerate_ddp requires --micro-batch-size 1")
        if args.logits_mode != "action_positions":
            raise SystemExit("accelerate_ddp refuses to fall back from action-position logits")
        if args.gradient_checkpointing != "off":
            raise SystemExit("gradient checkpointing stays off until the locked stack is validated")
        if args.model_precision != "bf16" or args.dtype != "bfloat16":
            raise SystemExit("accelerate_ddp SFT requires BF16")
        if int(args.global_batch_size) < 1 or int(args.save_every) < 1:
            raise SystemExit("global-batch-size and save-every must be positive")
        if experiment != "student_onpolicy_sft" and args.stage == "all":
            raise SystemExit(
                "accelerate_ddp does not collect and train in one GPU job. "
                "Seal the dataset, then relaunch with --stage train."
            )
    return args
