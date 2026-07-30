#!/usr/bin/env python3
"""Round 5 B4 — train one variant on full 1807/522."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.dup_diagnostics import write_json
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig

VARIANTS = {
    "o7_r64": {
        "loss_mode": "discriminative_ce",
        "compact_target": False,
        "lora_rank": 64,
        "lora_alpha": 128,
    },
    "compact_json": {
        "loss_mode": "sample_normalized_action_ce",
        "compact_target": True,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--train", type=Path,
                   default=_REPO / "artifacts/datasets/dup_sdi_round3/train.jsonl")
    p.add_argument("--valid", type=Path,
                   default=_REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO / "outputs/scope_round5/b4_full")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    vk = VARIANTS[args.variant]
    out = args.output_dir / f"{args.variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = SDITrainConfig(
        model_path=args.base_model,
        output_dir=out / "adapter",
        loss_mode=vk["loss_mode"],
        compact_target=vk["compact_target"],
        class_balancing=False,
        route_balancing=False,
        kl_coef=0.0,
        num_epochs=args.epochs,
        learning_rate=2e-5,
        batch_size=4,
        grad_accum=4,
        max_length=4096,
        lora_rank=vk["lora_rank"],
        lora_alpha=vk["lora_alpha"],
        seed=args.seed,
        device=args.gpu,
    )

    t0 = time.time()
    trainer = DupSDITrainer(cfg)
    trainer.train(args.train, args.valid)
    from training.scope.dup_diagnostics import load_jsonl
    valid_rows = load_jsonl(args.valid)
    metrics = evaluate_capability(trainer, valid_rows)

    report = {
        "variant": args.variant,
        "seed": args.seed,
        "wall_clock_s": time.time() - t0,
        "adapter": str(out / "adapter"),
        "valid_metrics": metrics,
    }
    write_json(out / "train_report.json", report)
    (out / "DONE").write_text("1\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
