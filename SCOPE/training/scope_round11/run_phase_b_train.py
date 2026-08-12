#!/usr/bin/env python3
"""Round11 Phase B factorized train variants."""

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

from training.scope_round9.run_wave_b_train import merge_lora
from training.scope_round11.factorized_trainer import FactorizedRollbackTrainer, FactorizedTrainConfig

DATA = _REPO / "artifacts/datasets/scope_round10/hier_sdi"
OUT_DEFAULT = _REPO / "outputs/scope_round11/phase_b"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
VALID = DATA / "valid.jsonl"
PHASE_A_DECISION = _REPO / "outputs/scope_round11/phase_a_state_factorization/PHASE_A_DECISION.json"


def selected_view() -> str:
    if PHASE_A_DECISION.exists():
        d = json.loads(PHASE_A_DECISION.read_text(encoding="utf-8"))
        return str(d.get("selected_stage1_view") or "A3")
    return "A3"


def build_variants(main_view: str) -> dict[str, dict]:
    return {
        "factorized_main_seed42": {
            "seed": 42,
            "stage1_view": main_view,
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": False,
        },
        "factorized_main_seed43": {
            "seed": 43,
            "stage1_view": main_view,
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": False,
        },
        "factorized_main_seed44": {
            "seed": 44,
            "stage1_view": main_view,
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": False,
        },
        "factorized_state_only_seed42": {
            "seed": 42,
            "stage1_view": "A1",
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": False,
        },
        "factorized_full_stage1_seed42": {
            "seed": 42,
            "stage1_view": "A0",
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": False,
        },
        "factorized_compact_signal_seed42": {
            "seed": 42,
            "stage1_view": "A3" if main_view == "A2" else ("A2" if main_view != "A3" else "A3"),
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": False,
            # Prefer A2/A3 compact: if main is A3 use A2 as compact control, else A3.
            "force_compact": True,
        },
        "factorized_ckpt_listwise_seed42": {
            "seed": 42,
            "stage1_view": main_view,
            "checkpoint_loss": "listwise",
            "operation_only": False,
            "checkpoint_only": True,
        },
        "factorized_ckpt_pairwise_seed42": {
            "seed": 42,
            "stage1_view": main_view,
            "checkpoint_loss": "pairwise",
            "operation_only": False,
            "checkpoint_only": True,
        },
    }


def main() -> None:
    main_view = selected_view()
    variants = build_variants(main_view)
    # Compact control: always A2 if main!=A2 else A3
    if "force_compact" in variants["factorized_compact_signal_seed42"]:
        variants["factorized_compact_signal_seed42"]["stage1_view"] = "A2" if main_view != "A2" else "A3"
        variants["factorized_compact_signal_seed42"].pop("force_compact", None)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(variants))
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--stage1-view", default=None, help="override Phase-A selected view")
    args = p.parse_args()

    vk = dict(variants[args.variant])
    if args.stage1_view:
        vk["stage1_view"] = args.stage1_view
    out = args.out_root / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (
        not args.force_retrain
        and (out / "merged" / "config.json").exists()
        and (out / "train_only_report.json").exists()
    ):
        print(f"[skip-train] {args.variant}")
        return

    train_path = DATA / "train_p0_75.jsonl"
    t0 = time.time()
    cfg = FactorizedTrainConfig(
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
        operation_only=bool(vk["operation_only"]),
        checkpoint_only=bool(vk["checkpoint_only"]),
        stage1_view=vk["stage1_view"],
        checkpoint_loss=vk["checkpoint_loss"],
        max_listwise_candidates=8,
        disable_replan=True,
        use_class_weight=False,
        lambda_ckpt=1.0,
    )
    report = FactorizedRollbackTrainer(cfg).train(train_path, VALID if VALID.exists() else None)
    merged = merge_lora(out)
    wall = time.time() - t0
    full = {
        "variant": args.variant,
        "train_path": str(train_path),
        "stage1_view": vk["stage1_view"],
        "checkpoint_loss": vk["checkpoint_loss"],
        "operation_only": vk["operation_only"],
        "checkpoint_only": vk["checkpoint_only"],
        "phase_a_selected_view": main_view,
        "train_report": report,
        "merged_path": str(merged),
        "wall_clock_s": wall,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (out / "train_only_report.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: full[k] for k in full if k != "train_report"}, indent=2))


if __name__ == "__main__":
    main()
