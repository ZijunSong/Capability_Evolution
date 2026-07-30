#!/usr/bin/env python3
"""Round 4 Barrier 4: operation_ce overfit128 with diagnostic instrumentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.binary_operation_metrics import accumulate_from_pairs
from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl, write_json
from training.scope.eval_dup_capability import evaluate_capability, _parse_operation
from training.scope.losses import operation_balance_weights
from training.scope.operation_scorer import operation_ce_loss, score_operations
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def margin_stats(trainer: DupSDITrainer, samples: list[dict]) -> dict[str, Any]:
    margins: list[float] = []
    by_op: dict[str, list[float]] = {"KEEP_EVIDENCE": [], "SKIP_DUPLICATE": []}
    for s in samples:
        ct = compact_target_from_sample(s)
        if not ct:
            continue
        r = score_operations(
            trainer.model, trainer.tokenizer, trainer._state_text(s), device=trainer.device
        )
        m = r.scores[DupOperation.SKIP_DUPLICATE.value] - r.scores[DupOperation.KEEP_EVIDENCE.value]
        margins.append(m)
        by_op[ct.operation.value].append(m)

    def agg(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        xs_s = sorted(xs)
        n = len(xs)
        return {
            "mean": sum(xs) / n,
            "median": xs_s[n // 2],
            "n": n,
        }

    return {
        "margin_mean": sum(margins) / max(len(margins), 1),
        "margin_by_gold": {k: agg(v) for k, v in by_op.items()},
        "n_scored": len(margins),
    }


def class_loss_and_grad(
    trainer: DupSDITrainer, samples: list[dict]
) -> dict[str, Any]:
    """One forward-backward pass per class to estimate mean loss and grad norm."""
    trainer.model.train()
    out: dict[str, Any] = {}
    for op_name in (DupOperation.KEEP_EVIDENCE, DupOperation.SKIP_DUPLICATE):
        class_rows = [
            s
            for s in samples
            if compact_target_from_sample(s)
            and compact_target_from_sample(s).operation == op_name
        ]
        if not class_rows:
            continue
        losses: list[float] = []
        grad_norms: list[float] = []
        for s in class_rows[:16]:  # subsample for speed
            trainer.model.zero_grad(set_to_none=True)
            tgt = compact_target_from_sample(s).operation
            loss = operation_ce_loss(
                trainer.model,
                trainer.tokenizer,
                trainer._state_text(s),
                tgt,
                device=trainer.device,
            )
            losses.append(float(loss.detach().item()))
            loss.backward()
            gn = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    gn += float(p.grad.data.norm(2).item() ** 2)
            grad_norms.append(gn**0.5)
        out[op_name.value] = {
            "mean_loss": sum(losses) / len(losses),
            "mean_grad_norm": sum(grad_norms) / len(grad_norms),
            "n_probed": len(losses),
        }
    return out


def effective_weights_demo(ops: list[str], enabled: bool) -> dict[str, float]:
    w = operation_balance_weights(ops, enabled=enabled)
    keep_w = skip_w = 0.0
    n_keep = n_skip = 0
    for op, wi in zip(ops, w.tolist()):
        if op == DupOperation.KEEP_EVIDENCE.value:
            keep_w += wi
            n_keep += 1
        else:
            skip_w += wi
            n_skip += 1
    return {
        "class_balancing_enabled": enabled,
        "sum_weight_keep": keep_w,
        "sum_weight_skip": skip_w,
        "per_sample_keep": keep_w / max(n_keep, 1),
        "per_sample_skip": skip_w / max(n_skip, 1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round4_overfit128",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "outputs/scope_round4/overfit128",
    )
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--class-balancing", action="store_true", default=True)
    p.add_argument("--no-class-balancing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_path = args.dataset_dir / "train.jsonl"
    train_rows = load_jsonl(train_path)
    ops = [
        compact_target_from_sample(s).operation.value
        for s in train_rows
        if compact_target_from_sample(s)
    ]

    class_bal = args.class_balancing and not args.no_class_balancing
    cfg = SDITrainConfig(
        model_path=args.base_model,
        output_dir=args.output_dir / "adapter",
        loss_mode="operation_ce",
        compact_target=True,
        class_balancing=class_bal,
        num_epochs=args.epochs,
        learning_rate=2e-5,
        batch_size=4,
        grad_accum=4,
        max_length=4096,
        lora_rank=16,
        lora_alpha=32,
        seed=args.seed,
        device="cuda:0",
    )
    trainer = DupSDITrainer(cfg)

    report: dict[str, Any] = {
        "config": {
            "epochs": args.epochs,
            "class_balancing": class_bal,
            "n_train": len(train_rows),
            "seed": args.seed,
        },
        "effective_sample_weights": effective_weights_demo(ops, class_bal),
        "pre_train": {
            "margins": margin_stats(trainer, train_rows),
            "class_loss_grad": class_loss_and_grad(trainer, train_rows),
            "eval": evaluate_capability(trainer, train_rows),
        },
    }

    print("[overfit128] training...", flush=True)
    train_summary = trainer.train(train_path, train_path)
    report["train_summary"] = train_summary

    post_eval = evaluate_capability(trainer, train_rows)
    report["post_train"] = {
        "margins": margin_stats(trainer, train_rows),
        "class_loss_grad": class_loss_and_grad(trainer, train_rows),
        "eval": post_eval,
    }

    keep_rec = post_eval.get("KEEP_EVIDENCE", {}).get("recall", 0)
    skip_rec = post_eval.get("SKIP_DUPLICATE", {}).get("recall", 0)
    acc = post_eval.get("operation_accuracy", 0)
    report["pass_criteria"] = {
        "train_accuracy_gt_95pct": acc > 0.95,
        "keep_recall_gt_90pct": keep_rec > 0.90,
        "skip_recall_gt_90pct": skip_rec > 0.90,
        "B4_PASS": acc > 0.95 and keep_rec > 0.90 and skip_rec > 0.90,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "overfit128_report.json", report)
    print(json.dumps(report["pass_criteria"], indent=2))
    print(f"Wrote {args.output_dir / 'overfit128_report.json'}")


if __name__ == "__main__":
    main()
