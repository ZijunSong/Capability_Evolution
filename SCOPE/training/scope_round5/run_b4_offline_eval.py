#!/usr/bin/env python3
"""Round 5 B4 offline evaluation with extended metrics."""

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
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.operation_scorer import score_operations
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def margin_distribution(trainer, samples):
    margins_keep, margins_skip = [], []
    for s in samples:
        ct = compact_target_from_sample(s)
        if not ct:
            continue
        cid, curated = trainer._operation_context(s)
        r = score_operations(
            trainer.model, trainer.tokenizer, trainer._state_text(s),
            device=trainer.device, candidate_id=cid, curated_document_ids=curated,
        )
        m = r.scores[DupOperation.SKIP_DUPLICATE.value] - r.scores[DupOperation.KEEP_EVIDENCE.value]
        if ct.operation == DupOperation.KEEP_EVIDENCE:
            margins_keep.append(m)
        else:
            margins_skip.append(m)
    def q(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        return xs[int(p * (len(xs) - 1))]
    return {
        "mean_margin_KEEP": sum(margins_keep) / max(len(margins_keep), 1),
        "mean_margin_SKIP": sum(margins_skip) / max(len(margins_skip), 1),
        "margin_KEEP_q25": q(margins_keep, 0.25),
        "margin_SKIP_q75": q(margins_skip, 0.75),
        "n_KEEP": len(margins_keep),
        "n_SKIP": len(margins_skip),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--loss-mode", default="discriminative_ce")
    p.add_argument("--compact-target", action="store_true")
    p.add_argument("--valid", type=Path,
                   default=_REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--gpu", default="cuda:0")
    args = p.parse_args()

    cfg = SDITrainConfig(
        model_path=args.base_model,
        output_dir=args.adapter.parent,
        adapter_path=str(args.adapter),
        loss_mode=args.loss_mode,
        compact_target=args.compact_target,
        eval_only=True,
        device=args.gpu,
    )
    trainer = DupSDITrainer(cfg)
    rows = load_jsonl(args.valid)
    metrics = evaluate_capability(trainer, rows)
    margins = margin_distribution(trainer, rows)
    pred_ops = []
    for s in rows:
        pred = trainer._greedy_action(s)
        pred_ops.append((pred or {}).get("operation", "PARSE_FAIL"))

    report = {
        "variant": args.variant,
        "adapter": str(args.adapter),
        **metrics,
        "margins": margins,
        "prediction_distribution": dict(__import__("collections").Counter(pred_ops)),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
