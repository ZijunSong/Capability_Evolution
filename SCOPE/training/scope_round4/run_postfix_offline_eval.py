#!/usr/bin/env python3
"""Round 4 Barrier 3: postfix offline capability eval (fixed Path handling)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.dup_diagnostics import load_jsonl
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig

VARIANTS: dict[str, dict] = {
    "Base": {"model_path": None, "loss_mode": "operation_ce"},
    "round3_compact_json": {
        "subdir": "round3_compact_json_sample_norm",
        "loss_mode": "sample_normalized_action_ce",
    },
    "round3_op_seed42": {"subdir": "round3_op_main_seed42", "loss_mode": "operation_ce"},
    "round3_op_seed43": {"subdir": "round3_op_main_seed43", "loss_mode": "operation_ce"},
    "round3_op_seed44": {"subdir": "round3_op_main_seed44", "loss_mode": "operation_ce"},
    "round3_op_no_balance": {"subdir": "round3_op_no_balance", "loss_mode": "operation_ce"},
    "round3_correct_only": {"subdir": "round3_correct_only_op", "loss_mode": "operation_ce"},
    "round3_endorse_only": {"subdir": "round3_endorse_only_op", "loss_mode": "operation_ce"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--valid", type=Path, default=_REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl")
    p.add_argument("--merged-root", type=Path, default=_REPO / "outputs/scope_round3/merged")
    p.add_argument("--output-dir", type=Path, default=_REPO / "outputs/scope_round4/postfix_replay/offline")
    p.add_argument("--variant", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant not in VARIANTS:
        raise SystemExit(f"unknown variant: {args.variant}")

    # Use cuda:0 within CUDA_VISIBLE_DEVICES set by the launcher.
    cfg_def = VARIANTS[args.variant]
    if cfg_def.get("model_path") is None and args.variant == "Base":
        model_path = args.base_model
    else:
        subdir = cfg_def.get("subdir", args.variant)
        model_path = str(args.merged_root / subdir)

    valid = load_jsonl(args.valid)
    tcfg = SDITrainConfig(
        model_path=model_path,
        output_dir=Path(f"/tmp/r4_postfix_{args.variant}"),
        loss_mode=cfg_def["loss_mode"],
        compact_target=True,
        eval_only=True,
        device="cuda:0",
    )
    trainer = DupSDITrainer(tcfg)
    report = evaluate_capability(trainer, valid)
    report["variant"] = args.variant
    report["model_path"] = model_path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.variant}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"done {args.variant} macro_f1={report.get('macro_f1', 0):.4f}")


if __name__ == "__main__":
    main()
