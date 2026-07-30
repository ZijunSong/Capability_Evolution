#!/usr/bin/env python3
"""Round 5 B2 — objective math documentation and one-step real-model test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl, write_json
from training.scope.operation_objectives import objective_math_description, operation_loss
from training.scope.operation_scorer import score_operations
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def one_step_test(trainer: DupSDITrainer, sample: dict, target: DupOperation) -> dict:
    trainer.model.train()
    state = trainer._state_text(sample)
    cid, curated = trainer._operation_context(sample)
    pre = score_operations(
        trainer.model, trainer.tokenizer, state, device=trainer.device,
        candidate_id=cid, curated_document_ids=curated,
    )
    pre_margin = (
        pre.scores[DupOperation.SKIP_DUPLICATE.value]
        - pre.scores[DupOperation.KEEP_EVIDENCE.value]
    )

    loss = operation_loss(
        trainer.model,
        trainer.tokenizer,
        state,
        target,
        objective="discriminative_ce",
        device=trainer.device,
        candidate_id=cid,
        curated_document_ids=curated,
    )
    loss_before = float(loss.detach().item())
    trainer.model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(
        float(p.grad.norm().item() ** 2)
        for p in trainer.model.parameters()
        if p.grad is not None
    ) ** 0.5
    params_before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    opt = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad], lr=1e-3,
    )
    opt.step()

    delta_norm = sum(
        float((p.detach() - params_before[n]).norm().item() ** 2)
        for n, p in trainer.model.named_parameters()
        if p.requires_grad and n in params_before
    ) ** 0.5

    post = score_operations(
        trainer.model, trainer.tokenizer, state, device=trainer.device,
        candidate_id=cid, curated_document_ids=curated,
    )
    post_margin = (
        post.scores[DupOperation.SKIP_DUPLICATE.value]
        - post.scores[DupOperation.KEEP_EVIDENCE.value]
    )

    margin_delta = post_margin - pre_margin
    if target == DupOperation.KEEP_EVIDENCE:
        margin_ok = margin_delta < 0
    else:
        margin_ok = margin_delta > 0

    return {
        "target": target.value,
        "loss_before": loss_before,
        "loss_after": float(loss.detach().item()),
        "margin_before": pre_margin,
        "margin_after": post_margin,
        "margin_delta": margin_delta,
        "margin_direction_ok": margin_ok,
        "lora_grad_norm": grad_norm,
        "lora_param_delta_norm": delta_norm,
        "lora_updated": delta_norm > 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--dataset",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round5_nested/train_d2.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "outputs/scope_round5/b2_objective",
    )
    args = p.parse_args()

    rows = load_jsonl(args.dataset)
    keep_row = next(r for r in rows if compact_target_from_sample(r).operation == DupOperation.KEEP_EVIDENCE)
    skip_row = next(r for r in rows if compact_target_from_sample(r).operation == DupOperation.SKIP_DUPLICATE)

    math_doc = {k: objective_math_description(k) for k in (
        "operation_ce", "discriminative_ce", "pairwise_margin", "single_token",
        "discriminative_ce_sum", "discriminative_ce_mean",
    )}

    cfg = SDITrainConfig(
        model_path=args.base_model,
        output_dir=args.output_dir / "tmp_adapter",
        loss_mode="discriminative_ce",
        kl_coef=0.0,
        class_balancing=False,
        device="cuda:0",
    )
    trainer = DupSDITrainer(cfg)

    keep_res = one_step_test(trainer, keep_row, DupOperation.KEEP_EVIDENCE)
    # reload fresh trainer for SKIP to avoid interference
    trainer2 = DupSDITrainer(cfg)
    skip_res = one_step_test(trainer2, skip_row, DupOperation.SKIP_DUPLICATE)

    b2_pass = (
        keep_res["margin_direction_ok"]
        and skip_res["margin_direction_ok"]
        and keep_res["lora_updated"]
        and skip_res["lora_updated"]
    )

    report = {
        "math": math_doc,
        "keep_one_step": keep_res,
        "skip_one_step": skip_res,
        "B2_PASS": b2_pass,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "b2_report.json", report)
    (args.output_dir.parent / "B2_PASS").write_text(str(b2_pass) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
