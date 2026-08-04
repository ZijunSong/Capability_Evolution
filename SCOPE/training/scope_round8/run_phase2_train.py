#!/usr/bin/env python3
"""Train one Round 8 Phase 2 rollback variant."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.rollback_sdi_trainer import RollbackSDITrainer, RollbackTrainConfig

VARIANTS: dict[str, dict] = {
    "rollback_o7_seed42": {"seed": 42, "lora_rank": 64, "lora_alpha": 128},
    "rollback_o7_seed43": {"seed": 43, "lora_rank": 64, "lora_alpha": 128},
    "rollback_o7_seed44": {"seed": 44, "lora_rank": 64, "lora_alpha": 128},
    "stop_o7_seed42": {"seed": 42, "lora_rank": 64, "lora_alpha": 128, "endorse_only": True},
    "rollback_prompt_hint_distill": {"seed": 42, "lora_rank": 64, "lora_alpha": 128, "hint": True},
    "rollback_trajectory_imitation": {"seed": 42, "lora_rank": 64, "lora_alpha": 128, "trajectory": True},
    "rollback_correct_only": {"seed": 42, "lora_rank": 64, "lora_alpha": 128, "correct_only": True},
    "rollback_soft_replan_only": {"seed": 42, "lora_rank": 64, "lora_alpha": 128, "soft_replan": True},
    "rollback_endorse_only": {"seed": 42, "lora_rank": 64, "lora_alpha": 128, "endorse_only": True},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--train", type=Path, default=_REPO / "artifacts/datasets/scope_round8/rollback_sdi/train.jsonl")
    p.add_argument("--valid", type=Path, default=_REPO / "artifacts/datasets/scope_round8/rollback_sdi/valid.jsonl")
    p.add_argument("--output-dir", type=Path, default=_REPO / "outputs/scope_round8/phase2_training")
    args = p.parse_args()

    vk = VARIANTS[args.variant]
    out = args.output_dir / args.variant
    if (out / "DONE").exists():
        print(f"[skip] {args.variant} DONE exists")
        return

    cfg = RollbackTrainConfig(
        model_path="/data/ppnm/models/Qwen2.5-7B-Instruct",
        output_dir=out,
        loss_mode="discriminative_ce",
        kl_coef=0.0,
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=1,
        grad_accum=16,
        max_length=4096,
        lora_rank=vk["lora_rank"],
        lora_alpha=vk["lora_alpha"],
        seed=vk["seed"],
        device=args.gpu,
        route_filter="CORRECT" if vk.get("correct_only") else ("ENDORSE" if vk.get("endorse_only") else None),
        hint_distill=bool(vk.get("hint")),
        trajectory_imitation=bool(vk.get("trajectory")),
        soft_replan_only=bool(vk.get("soft_replan")),
    )

    t0 = time.time()
    trainer = RollbackSDITrainer(cfg)
    report = trainer.train(args.train, args.valid)
    report["variant"] = args.variant
    report["wall_clock_s"] = time.time() - t0
    (out / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "DONE").write_text("1\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
