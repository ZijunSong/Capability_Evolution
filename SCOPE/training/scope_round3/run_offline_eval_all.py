#!/usr/bin/env python3
"""Merge all Round3 LoRA adapters + offline capability eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.dup_diagnostics import load_jsonl
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig

VARIANTS: dict[str, dict] = {
    "round3_op_main_seed42": {"loss_mode": "operation_ce", "compact_target": True},
    "round3_op_main_seed43": {"loss_mode": "operation_ce", "compact_target": True},
    "round3_op_main_seed44": {"loss_mode": "operation_ce", "compact_target": True},
    "round3_compact_json_sample_norm": {
        "loss_mode": "sample_normalized_action_ce",
        "compact_target": True,
    },
    "round3_legacy_full_action_token_ce": {"loss_mode": "legacy_token_ce"},
    "round3_correct_only_op": {"loss_mode": "operation_ce", "compact_target": True},
    "round3_endorse_only_op": {"loss_mode": "operation_ce", "compact_target": True},
    "round3_op_no_balance": {"loss_mode": "operation_ce", "compact_target": True},
}


def merge_one(base: str, adapter: Path, out: Path) -> None:
    if (out / "config.json").exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(_REPO / "training/merge_lora_hf.py"),
            "--base-model",
            base,
            "--adapter",
            str(adapter),
            "--output",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--train-root", type=Path, default=_REPO / "outputs/scope_round3/training")
    p.add_argument("--merge-root", type=Path, default=_REPO / "outputs/scope_round3/merged")
    p.add_argument("--valid", type=Path, default=_REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl")
    p.add_argument("--output", type=Path, default=_REPO / "outputs/scope_round3/eval/offline_capability.json")
    args = p.parse_args()

    valid = load_jsonl(args.valid)
    report: dict = {}
    for name, cfg in VARIANTS.items():
        adapter = args.train_root / name
        merged = args.merge_root / name
        if not (adapter / "adapter_config.json").exists() and not (adapter / "train_summary.json").exists():
            continue
        merge_one(args.base_model, adapter, merged)
        tcfg = SDITrainConfig(
            model_path=args.base_model,
            output_dir=adapter,
            adapter_path=str(adapter),
            loss_mode=cfg.get("loss_mode", "operation_ce"),
            compact_target=bool(cfg.get("compact_target", False)),
        )
        trainer = DupSDITrainer(tcfg)
        report[name] = evaluate_capability(trainer, valid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"macro_f1": (v.get("KEEP_EVIDENCE", {}).get("f1", 0) + v.get("SKIP_DUPLICATE", {}).get("f1", 0)) / 2} for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
