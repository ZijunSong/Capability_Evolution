#!/usr/bin/env python3
"""Round 4 Barrier 1.1: offline operation metric audit (no training)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.dup_diagnostics import load_jsonl
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
VALID = _REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl"
ROUND2_MAIN = _REPO / "outputs/scope_round2/training/round2_main"
ROUND3_TRAIN = _REPO / "outputs/scope_round3/training"
ROUND3_MERGED = _REPO / "outputs/scope_round3/merged"

VARIANTS: dict[str, dict] = {
    "Base": {
        "adapter": None,
        "loss_mode": "operation_ce",
        "compact_target": False,
        "merged": False,
    },
    "Round2-main": {
        "adapter": ROUND2_MAIN,
        "loss_mode": "sample_normalized_action_ce",
        "compact_target": True,
        "merged": False,
    },
    "round3_op_seed42": {
        "name": "round3_op_main_seed42",
        "loss_mode": "operation_ce",
        "compact_target": True,
        "merged": True,
    },
    "round3_op_seed43": {
        "name": "round3_op_main_seed43",
        "loss_mode": "operation_ce",
        "compact_target": True,
        "merged": True,
    },
    "round3_op_seed44": {
        "name": "round3_op_main_seed44",
        "loss_mode": "operation_ce",
        "compact_target": True,
        "merged": True,
    },
    "round3_compact_json": {
        "name": "round3_compact_json_sample_norm",
        "loss_mode": "sample_normalized_action_ce",
        "compact_target": True,
        "merged": True,
    },
    "round3_legacy_token_ce": {
        "name": "round3_legacy_full_action_token_ce",
        "loss_mode": "legacy_token_ce",
        "compact_target": False,
        "merged": True,
    },
    "round3_correct_only": {
        "name": "round3_correct_only_op",
        "loss_mode": "operation_ce",
        "compact_target": True,
        "merged": True,
    },
    "round3_endorse_only": {
        "name": "round3_endorse_only_op",
        "loss_mode": "operation_ce",
        "compact_target": True,
        "merged": True,
    },
    "round3_op_no_balance": {
        "name": "round3_op_no_balance",
        "loss_mode": "operation_ce",
        "compact_target": True,
        "merged": True,
    },
}


def merge_lora(base: str, adapter: Path, out: Path) -> None:
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


def resolve_model_path(
    label: str, cfg: dict, merge_root: Path, base_model: str
) -> tuple[str, str | None]:
    if cfg.get("merged"):
        vname = cfg["name"]
        adapter = ROUND3_TRAIN / vname
        merged = merge_root / vname
        merge_lora(base_model, adapter, merged)
        return str(merged), None
    adapter = cfg.get("adapter")
    if adapter is not None:
        return base_model, str(adapter)
    return base_model, None


def eval_variant(
    label: str, cfg: dict, valid: list, merge_root: Path, base_model: str
) -> dict:
    model_path, adapter_path = resolve_model_path(label, cfg, merge_root, base_model)
    tcfg = SDITrainConfig(
        model_path=model_path,
        output_dir=Path(f"/tmp/r4_eval_{label}"),
        adapter_path=adapter_path,
        loss_mode=cfg.get("loss_mode", "operation_ce"),
        compact_target=bool(cfg.get("compact_target", False)),
        eval_only=True,
    )
    trainer = DupSDITrainer(tcfg)
    report = evaluate_capability(trainer, valid)
    report["variant"] = label
    report["model_path"] = model_path
    report["adapter_path"] = adapter_path
    report["loss_mode"] = cfg.get("loss_mode")
    return report


def write_audit_md(report: dict, out_md: Path) -> None:
    lines = [
        "# Offline Metric Audit (Round 4 Barrier 1.1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Models evaluated",
        "",
        "| Variant | n | acc | macro_f1 | bal_acc | KEEP f1 | SKIP f1 | TP_KEEP | FP_KEEP | FN_KEEP | TP_SKIP | FP_SKIP | FN_SKIP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in report["models"].items():
        lines.append(
            f"| {name} | {m.get('n_valid', 0)} "
            f"| {m.get('accuracy', 0):.3f} "
            f"| {m.get('macro_f1', 0):.3f} "
            f"| {m.get('balanced_accuracy', 0):.3f} "
            f"| {m.get('KEEP_EVIDENCE', {}).get('f1', 0):.3f} "
            f"| {m.get('SKIP_DUPLICATE', {}).get('f1', 0):.3f} "
            f"| {m.get('TP_KEEP', 0)} "
            f"| {m.get('FP_KEEP', 0)} "
            f"| {m.get('FN_KEEP', 0)} "
            f"| {m.get('TP_SKIP', 0)} "
            f"| {m.get('FP_SKIP', 0)} "
            f"| {m.get('FN_SKIP', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Unit test status",
            "",
            f"- binary_operation_metrics tests: **{report.get('unit_tests_passed', 'unknown')}**",
            "",
            "## B1_PASS criteria",
            "",
            f"- offline binary metrics validated: **{report.get('b1_offline_valid', False)}**",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--valid", type=Path, default=VALID)
    p.add_argument("--merge-root", type=Path, default=ROUND3_MERGED)
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "outputs/scope_round4/metric_audit/offline_eval_fixed.json",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_REPO / "outputs/scope_round4/metric_audit/OFFLINE_METRIC_AUDIT.md",
    )
    p.add_argument("--variants", nargs="*", default=None)
    args = p.parse_args()
    base_model = args.base_model

    # Run unit tests first
    test_rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_REPO / "tests/scope/test_binary_operation_metrics.py"),
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    unit_pass = test_rc.returncode == 0

    valid = load_jsonl(args.valid)
    models: dict = {}
    selected = args.variants or list(VARIANTS.keys())
    for label in selected:
        if label not in VARIANTS:
            print(f"[skip] unknown variant {label}")
            continue
        print(f"[eval] {label}")
        models[label] = eval_variant(label, VARIANTS[label], valid, args.merge_root, base_model)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_path": str(args.valid),
        "n_valid": len(valid),
        "unit_tests_passed": unit_pass,
        "unit_test_output": test_rc.stdout + test_rc.stderr,
        "b1_offline_valid": unit_pass,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_audit_md(report, args.report)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
