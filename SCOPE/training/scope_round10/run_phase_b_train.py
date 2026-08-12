#!/usr/bin/env python3
"""Round 10 Phase B: CONTINUE-boundary targeted train variants."""

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

from training.scope_round9.hier_rollback_trainer import HierRollbackTrainer, HierTrainConfig
from training.scope_round9.run_wave_b_train import merge_lora

DATA = _REPO / "artifacts/datasets/scope_round10/hier_sdi"
OUT_DEFAULT = _REPO / "outputs/scope_round10/phase_b"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
VALID = DATA / "valid.jsonl"

# GPU mapping matches 0807-todo1.md §8 (except threshold-only which does not train).
VARIANTS: dict[str, dict] = {
    "r10_main_noweight_seed42": {
        "seed": 42,
        "train": "train_p0_75.jsonl",
        "use_class_weight": False,
        "include_candidate_summary": True,
    },
    "r10_main_noweight_seed43": {
        "seed": 43,
        "train": "train_p0_75.jsonl",
        "use_class_weight": False,
        "include_candidate_summary": True,
    },
    "r10_main_noweight_seed44": {
        "seed": 44,
        "train": "train_p0_75.jsonl",
        "use_class_weight": False,
        "include_candidate_summary": True,
    },
    "r10_natural_prior_noweight_seed42": {
        "seed": 42,
        "train": "train_natural.jsonl",
        "use_class_weight": False,
        "include_candidate_summary": True,
    },
    "r10_balanced50_noweight_seed42": {
        "seed": 42,
        "train": "train_balanced50.jsonl",
        "use_class_weight": False,
        "include_candidate_summary": True,
    },
    "r10_p0_exact_repro_seed42": {
        "seed": 42,
        "train": "train_p0_75.jsonl",
        "use_class_weight": True,
        "include_candidate_summary": True,
    },
    "r10_stage1_state_only_seed42": {
        "seed": 42,
        "train": "train_p0_75.jsonl",
        "use_class_weight": False,
        "include_candidate_summary": False,
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()

    vk = VARIANTS[args.variant]
    out = args.out_root / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (
        not args.force_retrain
        and (out / "merged" / "config.json").exists()
        and (out / "train_only_report.json").exists()
    ):
        print(f"[skip-train] {args.variant}")
        return

    train_path = DATA / vk["train"]
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    t0 = time.time()
    cfg = HierTrainConfig(
        model_path=BASE_MODEL,
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
        operation_only=False,
        checkpoint_only=False,
        include_candidate_summary=bool(vk["include_candidate_summary"]),
        hint_distill=False,
        max_listwise_candidates=4,
        disable_replan=True,
        use_class_weight=bool(vk["use_class_weight"]),
    )
    report = HierRollbackTrainer(cfg).train(train_path, VALID if VALID.exists() else None)
    merged = merge_lora(out)
    wall = time.time() - t0
    full = {
        "variant": args.variant,
        "train_path": str(train_path),
        "use_class_weight": bool(vk["use_class_weight"]),
        "include_candidate_summary": bool(vk["include_candidate_summary"]),
        "train_report": report,
        "merged_path": str(merged),
        "wall_clock_s": wall,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO.parent, text=True
        ).strip(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (out / "train_only_report.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()
