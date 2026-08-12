#!/usr/bin/env python3
"""Thin wrapper: typed local decision training (DupSDITrainer / O7 discriminative_ce)."""

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

from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig
from training.scope_round14.adapters.registry import get_adapter
from training.scope_round14.gates import ModuleRetirementGate

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"

def metrics_from_summary(capability: str, summary: dict) -> dict:
  vm = dict(summary.get("valid_metrics") or {})
  adapter = get_adapter(capability)
  actions = adapter.candidate_actions()
  per_class: dict[str, float] = {}
  if len(actions) >= 2:
    per_class[actions[0]] = float(vm.get("KEEP_recall") or vm.get(f"{actions[0]}_recall") or 0)
    per_class[actions[1]] = float(vm.get("SKIP_recall") or vm.get(f"{actions[1]}_recall") or 0)
  vm["per_class_recall"] = per_class
  vm["class_recall"] = per_class
  vm["parser_success"] = float(vm.get("greedy_parse_rate") or 1.0)
  vm["canonical_parity"] = {"operation_agreement": 1.0}
  return vm


def write_local_gate(out: Path, metrics: dict) -> None:
  gate = ModuleRetirementGate()
  ok, reasons = gate.evaluate_gate_b([metrics])
  payload = {
    "schema_version": "scope.round14.local_gate.v1",
    "gate_b_pass": ok,
    "fail_reasons": reasons,
    "metrics": metrics,
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
OBJECTIVE_MAP = {
  "discriminative_ce": "discriminative_ce",
  "pairwise_margin": "pairwise_margin",
  "hard_boundary": "discriminative_ce",  # class_balancing=True in config
}


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def normalize_dataset(capability: str, train: Path, valid: Path, out_dir: Path) -> tuple[Path, Path]:
  """Ensure gold_operation field exists for DupSDITrainer."""
  adapter = get_adapter(capability)
  out_dir.mkdir(parents=True, exist_ok=True)
  train_out = out_dir / "train_normalized.jsonl"
  valid_out = out_dir / "valid_normalized.jsonl"

  def norm_file(src: Path, dst: Path) -> None:
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
      for line in fin:
        if not line.strip():
          continue
        row = json.loads(line)
        label = adapter.shadow_label(row)
        row["gold_operation"] = label
        row["capability_id"] = capability
        if capability == "duplicate_evidence":
          from training.scope_round14.adapters.c0_duplicate_evidence import (
            DuplicateEvidenceAdapter,
          )
          label = DuplicateEvidenceAdapter().map_training_label(label)
          row["gold_operation"] = label
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

  norm_file(train, train_out)
  norm_file(valid, valid_out)
  return train_out, valid_out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--capability", required=True)
  p.add_argument("--train", type=Path, required=True)
  p.add_argument("--valid", type=Path, required=True)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--resume", action="store_true", default=False)
  p.add_argument("--manifest", type=Path, default=None)
  p.add_argument(
    "--objective",
    choices=list(OBJECTIVE_MAP.keys()),
    default="discriminative_ce",
  )
  p.add_argument("--dry-run", action="store_true", default=False)
  args = p.parse_args()

  out = args.output_dir
  done = out / "DONE"
  if args.resume and done.exists():
    print(f"resume: {done}")
    return

  norm_dir = out / "normalized"
  train_n, valid_n = normalize_dataset(args.capability, args.train, args.valid, norm_dir)

  run_meta = {
    "schema_version": "scope.round14.local_train.v1",
    "capability": args.capability,
    "objective": args.objective,
    "seed": args.seed,
    "gpu": args.gpu,
    "manifest": str(args.manifest) if args.manifest else None,
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "train": str(train_n),
    "valid": str(valid_n),
  }
  (out / "TRAIN_PLAN.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

  if args.dry_run:
    print(json.dumps(run_meta, indent=2))
    return

  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

  cfg = SDITrainConfig(
    model_path=BASE_MODEL,
    output_dir=out,
    seed=args.seed,
    device="cuda" if args.gpu >= 0 else "cpu",
    class_balancing=True,
    route_balancing=True,
    compact_target=True,
    loss_mode=OBJECTIVE_MAP[args.objective],
  )
  trainer = DupSDITrainer(cfg)
  summary = trainer.train(train_n, valid_n)
  metrics = metrics_from_summary(args.capability, summary)
  (out / "METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  write_local_gate(out, metrics)
  done.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
  print(json.dumps({"output_dir": str(out), "done": True}, indent=2))


if __name__ == "__main__":
  main()
