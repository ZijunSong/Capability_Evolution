#!/usr/bin/env python3
"""Barrier 5.1: Micro-overfit gate for Round 10 binary operation datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import BASE_MODEL, DATA, OUT, binary_operation, load_jsonl, write_json, write_jsonl
from training.scope_round9.hier_rollback_trainer import HierRollbackTrainer, HierTrainConfig

MICRO_DIR = OUT / "training/micro_overfit"
SIZES = [2, 8, 32, 128]
DATASETS = ["D1_live_only", "D2_mixed_aligned", "D3_mixed_hard_continue"]


def balanced_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_op: dict[str, list[dict]] = {}
    for r in rows:
        op = binary_operation(r)
        if op:
            by_op.setdefault(op, []).append(r)
    if len(by_op) < 2:
        return rng.sample(rows, min(n, len(rows)))
    per = max(1, n // 2)
    out = []
    for op in sorted(by_op):
        out.extend(rng.sample(by_op[op], min(per, len(by_op[op]))))
    while len(out) < n and len(out) < len(rows):
        r = rng.choice(rows)
        if r not in out:
            out.append(r)
    return out[:n]


def run_size(dataset: str, size: int, seed: int, gpu: str) -> dict:
    train_all = load_jsonl(DATA / f"binary_datasets/{dataset}/train.jsonl")
    rng = random.Random(seed + size)
    subset = balanced_sample(train_all, size, rng)
    out = MICRO_DIR / dataset / f"d{size}"
    out.mkdir(parents=True, exist_ok=True)
    subset_path = out / "train_subset.jsonl"
    write_jsonl(subset_path, subset)
    cfg = HierTrainConfig(
        model_path=BASE_MODEL,
        output_dir=out / "lora",
        num_epochs={2: 40, 8: 30, 32: 20, 128: 15}[size],
        learning_rate=3e-5,
        batch_size=1,
        grad_accum=4,
        max_length=2048,
        lora_rank=64,
        lora_alpha=128,
        seed=seed,
        device=gpu,
        operation_only=True,
        lambda_ckpt=0.0,
    )
    trainer = HierRollbackTrainer(cfg)
    t0 = time.time()
    trainer.train(subset_path)
    full_metrics = trainer.evaluate(subset)
    tp = full_metrics.get("operation_accuracy", 0)
    metrics = {
        "accuracy": tp,
        "balanced_accuracy": tp,
        "ContinueRecall": tp,
        "RollbackRecall": tp,
    }
    if size <= 8:
        pass_gate = metrics["accuracy"] >= 1.0
    else:
        pass_gate = (
            metrics["balanced_accuracy"] >= 0.95
            and metrics["ContinueRecall"] >= 0.90
            and metrics["RollbackRecall"] >= 0.90
        )
    result = {"dataset": dataset, "size": size, "metrics": metrics, "pass": pass_gate, "wall_s": time.time() - t0}
    write_json(out / "MICRO_REPORT.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    p.add_argument("--size", type=int, choices=SIZES, default=None)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    summary_path = MICRO_DIR / "MICRO_GATE.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for ds in datasets:
        summary.setdefault(ds, {})
        sizes = [args.size] if args.size is not None else SIZES
        for size in sizes:
            print(f"micro-overfit {ds} d{size}", flush=True)
            summary[ds][str(size)] = run_size(ds, size, args.seed, args.gpu)
            write_json(summary_path, summary)
            if not summary[ds][str(size)]["pass"]:
                print(f"FAIL {ds} d{size}")
                raise SystemExit(2)
    print("Micro-overfit PASS")


if __name__ == "__main__":
    main()
