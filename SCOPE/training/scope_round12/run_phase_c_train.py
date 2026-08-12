#!/usr/bin/env python3
"""Train Round12 Phase C variants (full_stage1 seeds / canonical listwise seeds)."""

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

from training.scope_round11.factorized_trainer import FactorizedRollbackTrainer, FactorizedTrainConfig
from training.scope_round11.run_phase_b_train import merge_lora

TRAIN = _REPO / "artifacts" / "datasets" / "scope_round10" / "hier_sdi" / "train_p0_75.jsonl"
VALID = _REPO / "artifacts" / "datasets" / "scope_round10" / "hier_sdi" / "valid.jsonl"
BASE = "/data/ppnm/models/Qwen2.5-7B-Instruct"


def parse_variant(name: str) -> dict:
    if name.startswith("full_stage1_seed"):
        return {
            "seed": int(name.rsplit("seed", 1)[1]),
            "stage1_view": "A0",
            "checkpoint_loss": "pairwise",
            "checkpoint_only": False,
            "operation_only": False,
        }
    if name.startswith("ckpt_canonical_listwise_seed"):
        return {
            "seed": int(name.rsplit("seed", 1)[1]),
            "stage1_view": "A0",
            "checkpoint_loss": "listwise",
            "checkpoint_only": True,
            "operation_only": False,
        }
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()

    vk = parse_variant(args.variant)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (
        not args.force_retrain
        and (out / "merged" / "config.json").exists()
        and (out / "train_only_report.json").exists()
    ):
        print(f"[skip-train] {args.variant}")
        return

    t0 = time.time()
    cfg = FactorizedTrainConfig(
        model_path=BASE,
        output_dir=out / "lora",
        loss_mode="discriminative_ce",
        kl_coef=0.0,
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=1,
        grad_accum=16,
        max_length=1536,
        lora_rank=64,
        lora_alpha=128,
        seed=int(vk["seed"]),
        device=args.gpu,
        operation_only=bool(vk["operation_only"]),
        checkpoint_only=bool(vk["checkpoint_only"]),
        stage1_view=vk["stage1_view"],
        checkpoint_loss=vk["checkpoint_loss"],
        max_listwise_candidates=8,
        disable_replan=True,
        use_class_weight=False,
        lambda_ckpt=1.0,
    )
    report = FactorizedRollbackTrainer(cfg).train(TRAIN, VALID if VALID.exists() else None)
    merged = merge_lora(out)
    full = {
        "variant": args.variant,
        "train_path": str(TRAIN),
        "stage1_view": vk["stage1_view"],
        "checkpoint_loss": vk["checkpoint_loss"],
        "operation_only": vk["operation_only"],
        "checkpoint_only": vk["checkpoint_only"],
        "train_report": report,
        "merged_path": str(merged),
        "wall_clock_s": time.time() - t0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
    }
    (out / "train_only_report.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: full[k] for k in full if k != "train_report"}, indent=2))


if __name__ == "__main__":
    main()
