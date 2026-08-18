#!/usr/bin/env python3
"""Short format-aware LoRA recovery from CLEAN_FULL checkpoint."""

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--mask-mode", choices=["full", "format_aware"], default="format_aware")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="FR")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.train_jsonl)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    if args.n_samples and args.n_samples > 0:
        order = order[: args.n_samples]
    train_rows = [rows[i] for i in order]
    manifest = build_run_manifest(
        run_id=f"FORMAT-REPAIR-{args.tag}-s{args.seed}",
        stage="B_format_repair",
        command=["python", "scripts/run_format_repair_cell.py"],
        repo_root=REPO,
        output_dir=out,
        extra={
            "mask_mode": args.mask_mode,
            "seed": args.seed,
            "n_samples": len(train_rows),
            "adapter_path": args.adapter_path,
            "used_rl": False,
            "LOCAL_COMPAT_ONLY": True,
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)
    backend = CleanSFTTrainer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device_map=f"cuda:{args.gpu}",
        learning_rate=args.lr,
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
            if step % 10 == 0 or i == 0:
                write_status_live(
                    out / "STATUS_LIVE.md",
                    stage="B_format_repair",
                    run_id=manifest["run_id"],
                    n_expected=args.epochs * len(train_rows),
                    n_finished=step,
                    extra={"loss": stats["loss"], "epoch": ep},
                )
                _dump(
                    out / "progress.json",
                    {"step": step, "loss": stats["loss"], "elapsed_s": time.time() - t0},
                )
    ckpt = out / "lora_checkpoint"
    backend.save_pretrained(str(ckpt))
    summary = {
        "tag": args.tag,
        "mask_mode": args.mask_mode,
        "seed": args.seed,
        "n_train": len(train_rows),
        "mean_train_loss": sum(losses) / max(1, len(losses)),
        "n_train_steps": len(losses),
        "train_seconds": time.time() - t0,
        "checkpoint_lora": str(ckpt),
        "base_adapter": args.adapter_path,
        "finite_loss": all(x == x and abs(x) < 1e6 for x in losses[-20:]) if losses else False,
        "used_rl": False,
    }
    _dump(out / "summary.json", summary)
    write_run_manifest(
        out / "RUN_MANIFEST.json",
        finalize_run_manifest(manifest, exit_code=0, completed_shards=["train"]),
    )
    (out / "DONE").write_text("ok\n")
    del backend
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
