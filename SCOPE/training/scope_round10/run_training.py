#!/usr/bin/env python3
"""Barrier 5: Train one Round 10 live-aligned binary operation variant."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import BASE_MODEL, DATA, OUT
from training.scope_round9.hier_rollback_trainer import HierRollbackTrainer, HierTrainConfig

TRAIN_OUT = OUT / "training"

VARIANTS: dict[str, dict] = {
    "rollback_live_aligned_seed42": {"dataset": "D2_mixed_aligned", "seed": 42},
    "rollback_live_aligned_seed43": {"dataset": "D2_mixed_aligned", "seed": 43},
    "rollback_live_aligned_seed44": {"dataset": "D2_mixed_aligned", "seed": 44},
    "rollback_live_only_seed42": {"dataset": "D1_live_only", "seed": 42},
    "rollback_offline_only_binary_seed42": {"dataset": "D0_offline_only", "seed": 42},
    "rollback_hard_continue_seed42": {"dataset": "D3_mixed_hard_continue", "seed": 42},
    "rollback_source_token_seed42": {"dataset": "D4_source_token", "seed": 42},
}


def merge_lora(variant_dir: Path) -> Path:
    merged = variant_dir / "merged"
    if (merged / "config.json").exists():
        return merged
    lora = variant_dir / "lora"
    if not (lora / "adapter_config.json").exists() and (lora / "lora" / "adapter_config.json").exists():
        lora = lora / "lora"
    merged.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            "training/merge_lora_hf.py",
            "--base-model",
            BASE_MODEL,
            "--adapter",
            str(lora),
            "--output",
            str(merged),
        ],
        cwd=_REPO,
        check=True,
    )
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--gpu", default="cuda:0")
    args = p.parse_args()

    vk = VARIANTS[args.variant]
    ds = DATA / "binary_datasets" / vk["dataset"]
    train_path = ds / "train.jsonl"
    valid_path = ds / "valid.jsonl"
    out = TRAIN_OUT / args.variant

    if (out / "merged" / "config.json").exists() and (out / "train_only_report.json").exists():
        print(f"[skip-train] {args.variant}")
        return

    t0 = time.time()
    cfg = HierTrainConfig(
        model_path=BASE_MODEL,
        output_dir=out / "lora",
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=1,
        grad_accum=16,
        max_length=2048,
        lora_rank=64,
        lora_alpha=128,
        seed=vk["seed"],
        device=args.gpu,
        operation_only=True,
        lambda_ckpt=0.0,
    )
    trainer = HierRollbackTrainer(cfg)
    report = trainer.train(train_path, valid_path)
    merged = merge_lora(out)
    full = {
        "variant": args.variant,
        "dataset": vk["dataset"],
        "train_report": report,
        "merged_path": str(merged),
        "wall_clock_s": time.time() - t0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
    }
    (out / "train_only_report.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()
