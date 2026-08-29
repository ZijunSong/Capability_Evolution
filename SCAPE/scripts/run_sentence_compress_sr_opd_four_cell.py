#!/usr/bin/env python3
"""sentence_compress formal 4-cell orchestrator (sr_opd_ce + CISPO).

This is the inventory entry the H100 audit asked for. It is not the
synthetic gpt-oss smoke runner and it does not reuse 20260821 reverse-KL
adapters.

Checklist:
  1. Read TRAIN_STATES_5K / event-active rows when present
  2. Student H_min snapshots; Teacher H_full side branch only
  3. sr_opd projection → Student-legal targets
  4. PURE_OPD = sr_opd_ce only (frozen states if available)
  5. RL = CISPO only
  6. RL+OPD = CISPO FB + CE FB + one optim_step
  7. Per-seed adapter + RUN_MANIFEST (formal seed 42)
  8. Adapter reload audit (manual safetensors)
  9. Generate actions on the BC+ 166-query test split
 10. Report Legal / Recall@5 / tool cost on the official 166-query test set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.training.four_cell_runtime import (
    build_manifest,
    coerce_runtime_args,
    run_seeded_four_cell,
    validate_wiring,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sentence_compress sr_opd_ce + CISPO four-cell")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--component", default="sentence_compress")
    p.add_argument("--training-mode", choices=("four_cell", "rl", "rl_opd", "pure_opd_only", "rl_opd_only"), default="four_cell")
    p.add_argument("--base-model", default="")
    p.add_argument("--sft-adapter", default="")
    p.add_argument("--gpu", default="0", help="GPU index or auto for model-parallel loading across visible GPUs")
    p.add_argument("--n-queries", type=int, default=664, help="Train queries. Formal RL / RL+OPD uses the full BC+ train split (664).")
    p.add_argument("--n-eval", type=int, default=None, help="Debug subset of the 166-query test split. Formal run leaves this unset.")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--lambda-opd", type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--opd-states-per-trajectory", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default="42", help="Comma-separated seeds. This formal rerun uses seed 42.")
    p.add_argument("--query-manifest", type=Path, default=None)
    p.add_argument("--eval-manifest", type=Path, default=None)
    p.add_argument("--train-states", type=Path, default=None, help="TRAIN_STATES_5K.jsonl or EVENT_ACTIVE_STATES_ALL.jsonl")
    p.add_argument("--n-train-states", type=int, default=None)
    p.add_argument("--official-eval", action="store_true", default=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rollout-backend", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--gpu-schedule", choices=("scheme_a", "resident_hf"), default="scheme_a")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--train-device-map", default="")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager")
    p.add_argument("--no-on-policy-refresh", action="store_false", dest="on_policy_refresh")
    p.add_argument("--on-policy-refresh", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    args = coerce_runtime_args(args)
    if args.component != "sentence_compress":
        raise SystemExit("this runner is sentence_compress-specific")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, extra={"runner": "run_sentence_compress_sr_opd_four_cell.py"})
    (args.out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.validate_only or args.dry_run:
        report = validate_wiring(args)
        (args.out / "VALIDATE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if not report.get("official_test_is_166"):
            raise SystemExit(f"official test split must be 166, got {report.get('official_test_count')}")
        if args.n_queries >= 664 and not report.get("using_full_train_split"):
            raise SystemExit(f"full train split must be 664, got {report.get('n_train_queries')}")
        if report.get("teacher_leak_in_student_prefix"):
            raise SystemExit("Teacher compressed view leaked into Student prefix")
        return 0
    if not args.base_model:
        raise SystemExit("--base-model is required for live training")
    summary = run_seeded_four_cell(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
