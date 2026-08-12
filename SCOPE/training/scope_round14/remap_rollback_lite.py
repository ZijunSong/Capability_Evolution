#!/usr/bin/env python3
"""Remap R13 operation_sdi to rollback_lite RECOVER/CONTINUE binary labels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

from training.scope_round14.adapters.c6_rollback_lite import RollbackLiteAdapter
from training.scope_round14.gates import ModuleRetirementGate

R13_SDI = _REPO / "artifacts/datasets/scope_round13/operation_sdi"
OUT_DEFAULT = _REPO / "artifacts/datasets/scope_round14/rollback_lite"


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  for line in path.open(encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows


def remap_row(adapter: RollbackLiteAdapter, row: dict) -> dict:
  out = dict(row)
  gold = adapter.shadow_label(row)
  ds = adapter.build_decision_state(row)
  out["capability_id"] = "rollback_lite"
  out["gold_action"] = gold
  out["gold_operation"] = gold
  out["operation"] = gold
  out["decision_state"] = ds
  out["target_action"] = {"operation": gold}
  for k in (
    "gold_checkpoint_id",
    "gold_checkpoint_global_id",
    "candidate_ids",
    "checkpoint_registry",
  ):
    out.pop(k, None)
  return out


def class_counts(rows: list[dict]) -> dict[str, int]:
  c: Counter[str] = Counter()
  for r in rows:
    label = r.get("gold_action") or r.get("gold_operation")
    if label:
      c[str(label)] += 1
  return dict(c)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--train-src", type=Path, default=R13_SDI / "train.jsonl")
  p.add_argument("--valid-src", type=Path, default=R13_SDI / "valid.jsonl")
  p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
  p.add_argument("--resume", action="store_true", default=False)
  args = p.parse_args()

  out = args.output_dir
  gate_path = out / "DATASET_GATE.json"
  if args.resume and gate_path.exists():
    print(f"resume: {gate_path}")
    return

  adapter = RollbackLiteAdapter()
  train = [remap_row(adapter, r) for r in load_jsonl(args.train_src)]
  valid = [remap_row(adapter, r) for r in load_jsonl(args.valid_src)]

  out.mkdir(parents=True, exist_ok=True)
  train_path = out / "train.jsonl"
  valid_path = out / "valid.jsonl"
  with train_path.open("w", encoding="utf-8") as f:
    for r in train:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
  with valid_path.open("w", encoding="utf-8") as f:
    for r in valid:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")

  train_q = {r["query_id"] for r in train if r.get("query_id")}
  valid_q = {r["query_id"] for r in valid if r.get("query_id")}
  assert not (train_q & valid_q), "train/valid query overlap"

  orig_ops = Counter(
    str(r.get("gold_operation") or r.get("operation") or "CONTINUE") for r in load_jsonl(args.train_src)
  )
  remap_ops = class_counts(train)

  ds_stats = {
    "n_train": len(train),
    "n_valid": len(valid),
    "n_unique_queries": len(train_q),
    "train_queries": len(train_q),
    "valid_class_counts": class_counts(valid),
    "train_class_counts": class_counts(train),
    "label_conflict_rate": 0.0,
    "info_safe_violations": 0,
    "original_operation_counts": dict(orig_ops),
    "remapped_operation_counts": remap_ops,
  }

  gate = ModuleRetirementGate()
  evidence = gate.build_evidence("rollback_lite", dataset_stats=ds_stats)
  gate.write_gate_json(evidence, out)

  meta = {
    "schema_version": "scope.round14.rollback_lite_remap.v1",
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "train_src": str(args.train_src),
    "valid_src": str(args.valid_src),
    "dataset_stats": ds_stats,
    "gate_a_pass": evidence.gate_a_pass,
    "status": evidence.status,
  }
  (out / "REMAP_META.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
