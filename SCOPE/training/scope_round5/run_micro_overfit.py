#!/usr/bin/env python3
"""Round 5 B3 — nested micro-overfit for one objective on one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl, write_json
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.operation_scorer import score_operations
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


OBJECTIVE_CONFIG = {
    "O0": {"loss_mode": "operation_ce"},
    "O1": {"loss_mode": "discriminative_ce"},
    "O2": {"loss_mode": "pairwise_margin"},
    "O3": {"loss_mode": "single_token"},
    "O4": {"loss_mode": "sample_normalized_action_ce", "compact_target": True},
    "O5": {"loss_mode": "discriminative_ce_sum"},
    "O6": {"loss_mode": "discriminative_ce_mean"},
    "O7": {"loss_mode": "discriminative_ce", "lora_rank": 64, "lora_alpha": 128},
}

SIZES = [2, 8, 32, 128]
GATES = {
    2: {"acc": 1.0, "keep_rec": 1.0, "skip_rec": 1.0},
    8: {"acc": 0.99, "keep_rec": 0.99, "skip_rec": 0.99},
    32: {"acc": 0.98, "keep_rec": 0.98, "skip_rec": 0.98},
    128: {"acc": 0.95, "keep_rec": 0.90, "skip_rec": 0.90},
}


def margin_stats(trainer: DupSDITrainer, samples: list[dict]) -> dict[str, Any]:
    margins_keep: list[float] = []
    margins_skip: list[float] = []
    for s in samples:
        ct = compact_target_from_sample(s)
        if not ct:
            continue
        r = score_operations(
            trainer.model, trainer.tokenizer, trainer._state_text(s), device=trainer.device,
            candidate_id=compact_target_from_sample(s).candidate_id,
            curated_document_ids=list(
                (s.get("decision_state") or {}).get("curated_document_ids")
                or (s.get("decision_state") or {}).get("curated_evidence_ids")
                or []
            ),
        )
        m = r.scores[DupOperation.SKIP_DUPLICATE.value] - r.scores[DupOperation.KEEP_EVIDENCE.value]
        if ct.operation == DupOperation.KEEP_EVIDENCE:
            margins_keep.append(m)
        else:
            margins_skip.append(m)
    return {
        "mean_margin_KEEP": sum(margins_keep) / max(len(margins_keep), 1),
        "mean_margin_SKIP": sum(margins_skip) / max(len(margins_skip), 1),
        "margin_separation_ok": (
            (sum(margins_skip) / max(len(margins_skip), 1) > 0)
            and (sum(margins_keep) / max(len(margins_keep), 1) < 0)
        ) if margins_keep and margins_skip else False,
    }


def epochs_for_size(size: int) -> int:
    return {2: 30, 8: 20, 32: 15, 128: 10}[size]


def run_one_size(
    objective: str,
    size: int,
    *,
    base_model: str,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    gpu: str,
) -> dict[str, Any]:
    cfg_kwargs = dict(OBJECTIVE_CONFIG[objective])
    compact = cfg_kwargs.pop("compact_target", False)
    lora_rank = cfg_kwargs.pop("lora_rank", 16)
    lora_alpha = cfg_kwargs.pop("lora_alpha", 32)

    train_path = dataset_dir / f"train_d{size}.jsonl"
    rows = load_jsonl(train_path)
    out = output_dir / f"d{size}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = SDITrainConfig(
        model_path=base_model,
        output_dir=out / "adapter",
        loss_mode=cfg_kwargs["loss_mode"],
        compact_target=compact,
        class_balancing=False,
        route_balancing=False,
        kl_coef=0.0,
        num_epochs=epochs_for_size(size),
        learning_rate=2e-5,
        batch_size=min(4, size),
        grad_accum=max(1, 4 // max(size, 1)),
        max_length=4096,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        seed=seed,
        device=gpu,
    )
    t0 = time.time()
    trainer = DupSDITrainer(cfg)
    pre_eval = evaluate_capability(trainer, rows)
    trainer.train(train_path, train_path)
    post_eval = evaluate_capability(trainer, rows)
    margins = margin_stats(trainer, rows)

    keep_rec = post_eval.get("KEEP_EVIDENCE", {}).get("recall", 0)
    skip_rec = post_eval.get("SKIP_DUPLICATE", {}).get("recall", 0)
    acc = post_eval.get("operation_accuracy", 0)
    gate = GATES[size]
    passed = (
        acc >= gate["acc"]
        and keep_rec >= gate["keep_rec"]
        and skip_rec >= gate["skip_rec"]
        and margins.get("margin_separation_ok", False)
    )

    report = {
        "objective": objective,
        "size": size,
        "n_train": len(rows),
        "wall_clock_s": time.time() - t0,
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "margins": margins,
        "gate": gate,
        "passed": passed,
    }
    write_json(out / "report.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objective", required=True, choices=list(OBJECTIVE_CONFIG))
    p.add_argument("--dataset-dir", type=Path,
                   default=_REPO / "artifacts/datasets/dup_sdi_round5_nested")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO / "outputs/scope_round5/micro_overfit")
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="cuda:0")
    args = p.parse_args()

    obj_dir = args.output_dir / args.objective
    obj_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"objective": args.objective, "sizes": {}}
    all_pass = True

    for size in SIZES:
        print(f"[{args.objective}] D{size} ...", flush=True)
        rep = run_one_size(
            args.objective, size,
            base_model=args.base_model,
            dataset_dir=args.dataset_dir,
            output_dir=obj_dir,
            seed=args.seed,
            gpu=args.gpu,
        )
        results["sizes"][f"D{size}"] = rep
        if not rep["passed"]:
            all_pass = False
            print(f"[{args.objective}] D{size} FAIL — stopping cascade", flush=True)
            break
        print(f"[{args.objective}] D{size} PASS", flush=True)

    results["all_pass"] = all_pass
    write_json(obj_dir / "summary.json", results)
    (obj_dir / ("PASS" if all_pass else "FAIL")).write_text("1\n")
    print(json.dumps({"objective": args.objective, "all_pass": all_pass}, indent=2))


if __name__ == "__main__":
    main()
