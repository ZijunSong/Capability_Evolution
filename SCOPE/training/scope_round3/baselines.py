#!/usr/bin/env python3
"""Offline baselines B0/B1/B2 for Round 3 valid set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def majority_baseline(samples: list[dict], train_samples: list[dict]) -> dict:
    ops = []
    for s in train_samples:
        ct = compact_target_from_sample(s)
        if ct:
            ops.append(ct.operation.value)
    majority = Counter(ops).most_common(1)[0][0] if ops else DupOperation.SKIP_DUPLICATE.value
    tp_keep = tp_skip = fp_keep = fp_skip = fn_keep = fn_skip = 0
    for s in samples:
        ct = compact_target_from_sample(s)
        if not ct:
            continue
        pred = majority
        tgt = ct.operation.value
        if tgt == DupOperation.KEEP_EVIDENCE.value:
            if pred == tgt:
                tp_keep += 1
            else:
                fn_keep += 1
            if pred == DupOperation.SKIP_DUPLICATE.value:
                fp_keep += 1
        else:
            if pred == tgt:
                tp_skip += 1
            else:
                fn_skip += 1
            if pred == DupOperation.KEEP_EVIDENCE.value:
                fp_skip += 1
    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        return {"precision": p, "recall": r, "f1": f1}
    keep = prf(tp_keep, fp_keep, fn_keep)
    skip = prf(tp_skip, fp_skip, fn_skip)
    return {
        "name": "B0_majority",
        "majority_class": majority,
        "KEEP_EVIDENCE": keep,
        "SKIP_DUPLICATE": skip,
        "macro_f1": (keep["f1"] + skip["f1"]) / 2,
        "balanced_accuracy": (keep["recall"] + skip["recall"]) / 2,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--round2-adapter", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    valid = load_jsonl(args.valid)
    train = load_jsonl(args.train)
    report = {"B0_majority": majority_baseline(valid, train)}

    cfg_base = SDITrainConfig(
        model_path=args.model_path,
        output_dir=Path("/tmp/r3_base_eval"),
        loss_mode="operation_ce",
    )
    trainer_base = DupSDITrainer(cfg_base)
    report["B1_base_operation_ce"] = {
        "name": "B1_base",
        **evaluate_capability(trainer_base, valid),
    }

    if args.round2_adapter:
        cfg_r2 = SDITrainConfig(
            model_path=args.model_path,
            output_dir=Path("/tmp/r3_r2_eval"),
            adapter_path=str(args.round2_adapter),
            compact_target=True,
            loss_mode="sample_normalized_action_ce",
        )
        trainer_r2 = DupSDITrainer(cfg_r2)
        report["B2_round2_main"] = {
            "name": "B2_round2",
            **evaluate_capability(trainer_r2, valid),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
