#!/usr/bin/env python3
"""Train one Round 9 Wave B hierarchical rollback variant."""

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
from training.scope.rollback_sdi_trainer import RollbackSDITrainer, RollbackTrainConfig

VARIANTS: dict[str, dict] = {
    "rollback_hier_o7_seed42": {"seed": 42, "hier": True},
    "rollback_hier_o7_seed43": {"seed": 43, "hier": True},
    "rollback_hier_o7_seed44": {"seed": 44, "hier": True},
    "rollback_flat_o7_seed42_repro": {"seed": 42, "flat": True},
    "rollback_operation_only_seed42": {"seed": 42, "hier": True, "operation_only": True},
    "rollback_checkpoint_ranker_seed42": {"seed": 42, "hier": True, "checkpoint_only": True},
    "rollback_hier_no_candidate_summary_seed42": {"seed": 42, "hier": True, "no_summary": True},
    "rollback_hier_prompt_hint_seed42": {"seed": 42, "hier": True, "hint": True},
}

TRAIN = _REPO / "artifacts/datasets/scope_round9/hier_sdi/train.jsonl"
VALID = _REPO / "artifacts/datasets/scope_round9/hier_sdi/valid.jsonl"
HOLDOUT = _REPO / "artifacts/datasets/scope_round9/hier_sdi/frozen_live_holdout.jsonl"
OUT_DEFAULT = _REPO / "outputs/scope_round9/wave_b"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"


def merge_lora(variant_dir: Path) -> Path:
    merged = variant_dir / "merged"
    if (merged / "config.json").exists():
        return merged
    lora = variant_dir / "lora"
    # Tolerate accidental nested save (.../lora/lora) from earlier trainer bug.
    if not (lora / "adapter_config.json").exists() and (lora / "lora" / "adapter_config.json").exists():
        lora = lora / "lora"
    if not (lora / "adapter_config.json").exists():
        raise FileNotFoundError(f"missing LoRA adapter at {variant_dir / 'lora'}")
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
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Wave B output root (default: outputs/scope_round9/wave_b)",
    )
    p.add_argument(
        "--disable-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="P0: binary CONTINUE vs ROLLBACK_TO (default: true)",
    )
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()

    vk = VARIANTS[args.variant]
    out_root = args.out_root or OUT_DEFAULT
    out = out_root / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (
        not args.force_retrain
        and (out / "merged" / "config.json").exists()
        and (out / "train_only_report.json").exists()
    ):
        print(f"[skip-train] {args.variant} already merged under {out_root}")
        print((out / "train_only_report.json").read_text(encoding="utf-8"))
        return

    # Resume path: LoRA already saved (possibly nested) but merge failed earlier.
    lora_root = out / "lora"
    lora_ready = (lora_root / "adapter_config.json").exists() or (
        lora_root / "lora" / "adapter_config.json"
    ).exists()
    if (
        not args.force_retrain
        and lora_ready
        and not (out / "merged" / "config.json").exists()
    ):
        print(f"[resume-merge] {args.variant}: merging existing LoRA without retrain", flush=True)
        t0 = time.time()
        merged = merge_lora(out)
        wall = time.time() - t0
        full_report = {
            "variant": args.variant,
            "train_report": {"note": "resumed merge from existing LoRA; training skipped"},
            "merged_path": str(merged),
            "wall_clock_s": wall,
            "recovered": True,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
            ).strip(),
        }
        (out / "train_only_report.json").write_text(
            json.dumps(full_report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(full_report, indent=2))
        return

    t0 = time.time()
    if vk.get("flat"):
        cfg = RollbackTrainConfig(
            model_path=BASE_MODEL,
            output_dir=out / "lora",
            loss_mode="discriminative_ce",
            kl_coef=0.0,
            num_epochs=3,
            learning_rate=2e-5,
            batch_size=1,
            grad_accum=16,
            max_length=2048,
            lora_rank=64,
            lora_alpha=128,
            seed=vk["seed"],
            device=args.gpu,
        )
        report = RollbackSDITrainer(cfg).train(TRAIN, VALID)
    else:
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
            seed=vk["seed"],
            device=args.gpu,
            operation_only=bool(vk.get("operation_only")),
            checkpoint_only=bool(vk.get("checkpoint_only")),
            include_candidate_summary=not vk.get("no_summary"),
            hint_distill=bool(vk.get("hint")),
            max_listwise_candidates=4,
            disable_replan=bool(args.disable_replan),
        )
        report = HierRollbackTrainer(cfg).train(TRAIN, VALID)

    merged = merge_lora(out)
    wall = time.time() - t0
    full_report = {
        "variant": args.variant,
        "train_report": report,
        "merged_path": str(merged),
        "wall_clock_s": wall,
        "p0": {
            "disable_replan": bool(args.disable_replan),
            "out_root": str(out_root),
            "train_path": str(TRAIN),
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
    }
    (out / "train_only_report.json").write_text(
        json.dumps(full_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(full_report, indent=2))


if __name__ == "__main__":
    main()
