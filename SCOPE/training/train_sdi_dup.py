#!/usr/bin/env python3
"""Dup-only SDI Round-1 training entry (action CE + KL stabilization)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs/scope/sdi_dup_only.yaml",
    )
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--loss-mode", type=str, default=None)
    p.add_argument("--route-balancing", action="store_true", default=None)
    p.add_argument("--compact-target", action="store_true", default=None)
    p.add_argument("--route-filter", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--class-balancing", action="store_true", default=None)
    p.add_argument("--sequence-ce-coef", type=float, default=None)
    p.add_argument("--operation-ce-coef", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scope = cfg_raw.get("scope") or {}
    sdi = scope.get("sdi") or {}
    train_cfg = scope.get("train") or {}
    dataset = scope.get("dataset") or {}

    dataset_dir = Path(dataset.get("dir", _REPO_ROOT / "artifacts/datasets/dup_sdi_round1"))
    train_path = args.train or dataset_dir / "train.jsonl"
    valid_path = args.valid or dataset_dir / "valid.jsonl"
    output_dir = args.output_dir or Path(
        train_cfg.get("output_dir", _REPO_ROOT / "outputs/dup_sdi_round1")
    )

    cfg = SDITrainConfig(
        model_path=args.model_path
        or train_cfg.get("model_path", "/data/ppnm/models/Qwen2.5-7B-Instruct"),
        output_dir=output_dir,
        kl_coef=float(sdi.get("kl_coef", 0.01)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_epochs=int(train_cfg.get("num_epochs", 3)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        grad_accum=int(train_cfg.get("grad_accum", 4)),
        max_length=int(train_cfg.get("max_length", 4096)),
        lora_rank=int(
            args.lora_rank
            if args.lora_rank is not None
            else train_cfg.get("lora_rank", 16)
        ),
        lora_alpha=int(
            args.lora_alpha
            if args.lora_alpha is not None
            else train_cfg.get("lora_alpha", 32)
        ),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        device=str(train_cfg.get("device", "cuda")),
        loss_mode=args.loss_mode or sdi.get("loss_mode", "sample_normalized_action_ce"),
        route_balancing=(
            args.route_balancing
            if args.route_balancing is not None
            else bool(sdi.get("route_balancing", False))
        ),
        compact_target=(
            args.compact_target
            if args.compact_target is not None
            else bool(sdi.get("compact_target", False))
        ),
        route_filter=args.route_filter or sdi.get("route_filter"),
        seed=args.seed if args.seed is not None else int(train_cfg.get("seed", 42)),
        class_balancing=(
            args.class_balancing
            if args.class_balancing is not None
            else bool(sdi.get("class_balancing", False))
        ),
        sequence_ce_coef=float(
            args.sequence_ce_coef
            if args.sequence_ce_coef is not None
            else sdi.get("sequence_ce_coef", 1.0)
        ),
        operation_ce_coef=float(
            args.operation_ce_coef
            if args.operation_ce_coef is not None
            else sdi.get("operation_ce_coef", 1.0)
        ),
        max_steps=(
            args.max_steps
            if args.max_steps is not None
            else train_cfg.get("max_steps")
        ),
    )

    trainer = DupSDITrainer(cfg)
    summary = trainer.train(train_path, valid_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
