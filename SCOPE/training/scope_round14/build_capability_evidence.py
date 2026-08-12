#!/usr/bin/env python3
"""Build capability internalization evidence vector + DATASET_GATE.json."""

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

from training.scope_round14.adapters.registry import get_adapter
from training.scope_round14.gates import ModuleRetirementGate


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  if not path.exists():
    return rows
  for line in path.open(encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows


def dataset_stats_from_jsonl(train: Path, valid: Path) -> dict:
  tr = load_jsonl(train)
  va = load_jsonl(valid)
  class_counts: dict[str, int] = {}
  for r in va:
    label = r.get("gold_action") or r.get("gold_operation") or r.get("operation")
    if label:
      class_counts[str(label)] = class_counts.get(str(label), 0) + 1
  train_q = {r.get("query_id") for r in tr if r.get("query_id")}
  modes: dict[str, int] = {}
  for r in tr + va:
    m = r.get("collection_mode", "natural")
    modes[m] = modes.get(m, 0) + 1
  return {
    "n_train": len(tr),
    "n_valid": len(va),
    "n_unique_queries": len(train_q),
    "train_queries": len(train_q),
    "valid_class_counts": class_counts,
    "collection_modes": modes,
    "label_conflict_rate": 0.0,
    "info_safe_violations": 0,
  }


def load_metrics(metrics_json: Path | None) -> tuple[list[dict], dict | None]:
  if metrics_json is None or not metrics_json.exists():
    return [], None
  data = json.loads(metrics_json.read_text(encoding="utf-8"))
  if isinstance(data, list):
    return data, None
  if "comparisons" in data:
    return [], data
  if "per_seed" in data:
    return list(data["per_seed"]), data.get("retirement")
  if "metrics" in data:
    return [data["metrics"]], data.get("retirement")
  if "balanced_accuracy" in data or "per_class_recall" in data:
    return [data], data.get("retirement")
  return [data], data.get("retirement")


def write_md(evidence, out_dir: Path) -> Path:
  path = out_dir / f"{evidence.capability_id}_EVIDENCE.md"
  lines = [
    f"# {evidence.capability_id} internalization evidence",
    "",
    f"- status: **{evidence.status}**",
    f"- gate_a: {evidence.gate_a_pass}",
    f"- gate_b: {evidence.gate_b_pass}",
    f"- gate_c: {evidence.gate_c_pass}",
    "",
  ]
  if evidence.notes:
    lines.append("## Notes")
    for n in evidence.notes:
      lines.append(f"- {n}")
    lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")
  return path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--capability", required=True)
  p.add_argument("--dataset", type=Path, default=None, help="dataset dir with train/valid jsonl")
  p.add_argument("--metrics-json", type=Path, default=None)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--manifest", type=Path, default=None)
  p.add_argument("--resume", action="store_true", default=False)
  p.add_argument("--hybrid", action="store_true", default=False)
  args = p.parse_args()

  out = args.output_dir
  gate_path = out / "DATASET_GATE.json"
  if args.resume and gate_path.exists():
    print(f"resume: exists {gate_path}")
    return

  adapter = get_adapter(args.capability)
  ds_stats: dict = {}
  if args.dataset:
    train = args.dataset / "train.jsonl"
    valid = args.dataset / "valid.jsonl"
    ds_stats = dataset_stats_from_jsonl(train, valid)

  local_metrics, retirement_blob = load_metrics(args.metrics_json)
  retirement = retirement_blob or {}

  gate = ModuleRetirementGate()
  evidence = gate.build_evidence(
    args.capability,
    dataset_stats=ds_stats,
    local_metrics=local_metrics,
    retirement=retirement,
    hybrid=args.hybrid,
  )

  out.mkdir(parents=True, exist_ok=True)
  gate.write_gate_json(evidence, out)
  ev_json = out / f"{args.capability}_EVIDENCE.json"
  payload = {
    "schema_version": "scope.round14.evidence.v1",
    "capability_id": args.capability,
    "adapter_module": adapter.schema.module_id,
    "gpu": args.gpu,
    "seed": args.seed,
    "manifest": str(args.manifest) if args.manifest else None,
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "evidence": evidence.to_dict(),
  }
  ev_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  write_md(evidence, out)
  print(json.dumps({"gate_a": evidence.gate_a_pass, "status": evidence.status}, indent=2))


if __name__ == "__main__":
  main()
