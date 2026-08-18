#!/usr/bin/env python3
"""One Clean-SFT cell: smoke then optional full public SFT (FULL or TOOL mask)."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.status import write_status_live
from scape.training.clean_sft import CleanSFTTrainer, load_jsonl


def _dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.train_jsonl))
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    if args.n_samples and args.n_samples > 0:
        order = order[: args.n_samples]
    train_rows = [rows[i] for i in order]
    manifest = build_run_manifest(
        run_id=f"CLEAN-SFT-{args.mask_mode}-s{args.seed}-n{len(train_rows)}",
        stage="C0_clean_sft",
        command=["python", "scripts/run_clean_sft_cell.py"],
        repo_root=REPO,
        output_dir=out,
        extra={
            "mask_mode": args.mask_mode,
            "seed": args.seed,
            "n_samples": len(train_rows),
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lr": args.lr,
            "max_length": args.max_length,
            "base_model": args.model_path,
            "legacy_scope_path_used": False,
            "LOCAL_COMPAT_ONLY": True,
            "used_rl": False,
            "used_released_harness1_ckpt": False,
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)
    write_status_live(
        out / "STATUS_LIVE.md",
        stage="C0_clean_sft",
        run_id=manifest["run_id"],
        n_expected=max(1, args.epochs * len(train_rows)),
        n_finished=0,
        extra={"mask_mode": args.mask_mode, "phase": "load_model"},
    )
    device_map = f"cuda:{args.gpu}" if args.gpu is not None else "auto"
    backend = CleanSFTTrainer(
        model_path=args.model_path,
        device_map=device_map,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r,
        max_length=args.max_length,
        mask_mode=args.mask_mode,
    )
    losses: list[float] = []
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for i, ex in enumerate(train_rows):
            stats = backend.train_step([ex])
            step += 1
            if stats["n"] > 0:
                losses.append(stats["loss"])
            if step % 20 == 0 or i == 0:
                write_status_live(
                    out / "STATUS_LIVE.md",
                    stage="C0_clean_sft",
                    run_id=manifest["run_id"],
                    n_expected=args.epochs * len(train_rows),
                    n_finished=step,
                    extra={
                        "epoch": ep,
                        "loss": stats["loss"],
                        "mask_mode": args.mask_mode,
                        "lora_targets": backend.lora_targets,
                    },
                )
                _dump(
                    out / "progress.json",
                    {
                        "step": step,
                        "epoch": ep,
                        "loss": stats["loss"],
                        "elapsed_s": time.time() - t0,
                    },
                )
    train_s = time.time() - t0
    ckpt = out / "lora_checkpoint"
    backend.save_pretrained(str(ckpt))
    merged = None
    if args.merge:
        merged = out / "hf_merged"
        backend.merge_and_save(str(merged))
    summary = {
        "mask_mode": args.mask_mode,
        "seed": args.seed,
        "n_train": len(train_rows),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lr": args.lr,
        "max_length": args.max_length,
        "mean_train_loss": sum(losses) / max(1, len(losses)),
        "n_train_steps": len(losses),
        "train_seconds": train_s,
        "lora_targets": backend.lora_targets,
        "checkpoint_lora": str(ckpt),
        "checkpoint_merged": str(merged) if merged else None,
        "base_model": args.model_path,
        "legacy_scope_path_used": False,
        "LOCAL_COMPAT_ONLY": True,
        "used_rl": False,
        "used_released_harness1_ckpt": False,
    }
    _dump(out / "summary.json", summary)
    write_status_live(
        out / "STATUS_LIVE.md",
        stage="C0_clean_sft",
        run_id=manifest["run_id"],
        n_expected=args.epochs * len(train_rows),
        n_finished=args.epochs * len(train_rows),
        extra={"mean_train_loss": summary["mean_train_loss"]},
    )
    write_run_manifest(
        out / "RUN_MANIFEST.json",
        finalize_run_manifest(manifest, exit_code=0, completed_shards=["train"]),
    )
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    del backend
    gc.collect()
    try:
        import torch

        if args.gpu is not None:
            with torch.cuda.device(args.gpu):
                torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--mask-mode", choices=["full", "tool"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    try:
        summary = run(args)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "FAILED.json").write_text(
            json.dumps({"error": str(exc)}, indent=2) + "\n"
        )
        print(json.dumps({"error": str(exc)}))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
