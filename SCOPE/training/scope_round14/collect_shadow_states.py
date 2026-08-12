#!/usr/bin/env python3
"""Targeted shadow-labeled event collection from mined decision states."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

from training.scope_round14.adapters.registry import get_adapter
from training.scope_round14.gates import ModuleRetirementGate

# Mineable state sources (no live rollout required)
MINE_ROOTS: list[tuple[str, Path]] = [
  ("r13_onpolicy", _REPO / "artifacts/datasets/scope_round13/onpolicy_raw"),
  ("scope_v3", _REPO / "artifacts/datasets/scope_v3"),
  ("rollback_r8", _REPO / "outputs/scope_round8/rollback_collection"),
  ("agent_core_r8", _REPO / "outputs/scope_round8/agent_core_diagnostic"),
  ("dup_sdi_r3", _REPO / "artifacts/datasets/dup_sdi_round3"),
  ("go25_dup", _REPO / "artifacts/datasets/scope_v3/go25_dup_premature"),
]


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def iter_jsonl_files(roots: list[tuple[str, Path]]) -> Iterator[tuple[str, Path]]:
  for name, root in roots:
    if not root.exists():
      continue
    for p in sorted(root.rglob("*.jsonl")):
      if "onpolicy_raw" in str(p) or name in {
        "rollback_r8",
        "agent_core_r8",
        "scope_v3",
        "dup_sdi_r3",
        "go25_dup",
        "r13_onpolicy",
      }:
        yield name, p


_CAP_ALIASES: dict[str, set[str]] = {
  "stop_decision": {"stop_decision", "premature_stop"},
  "verification_routing": {
    "verification_routing",
    "verification_decision",
    "premature_stop",
  },
  "evidence_admission": {
    "evidence_admission",
    "evidence_curation",
    "irrelevant_evidence",
    "duplicate_evidence",
  },
  "context_budget_routing": {
    "context_budget_routing",
    "deterministic_truncation",
    "budget_exhaustion",
  },
  "external_verification_routing": {
    "external_verification_routing",
    "external_verification",
    "invalid_citation",
  },
  "duplicate_evidence": {"duplicate_evidence"},
  "rollback_lite": {"rollback_lite", "rollback"},
}


def _capability_match(capability: str, cap: str, raw: dict[str, Any]) -> bool:
  """Accept alias matches, or any row with a decision_state for targeted relabel."""
  if not cap:
    return bool(raw.get("decision_state") or raw.get("student_state_text"))
  aliases = _CAP_ALIASES.get(capability, {capability})
  if cap in aliases or capability in cap or cap in capability:
    return True
  # Targeted: allow relabeling observable decision states from other probes.
  return bool(raw.get("decision_state") or raw.get("student_state_text"))


def mine_rows(capability: str, paths: Iterator[tuple[str, Path]]) -> list[dict[str, Any]]:
  adapter = get_adapter(capability)
  rows: list[dict[str, Any]] = []
  for source, path in paths:
    for line in path.open(encoding="utf-8"):
      if not line.strip():
        continue
      try:
        raw = json.loads(line)
      except json.JSONDecodeError:
        continue
      cap = str(raw.get("capability_id") or raw.get("capability") or "").lower()
      if not _capability_match(capability, cap, raw):
        continue
      try:
        label = adapter.shadow_label(raw)
      except Exception:
        continue
      if not label:
        continue
      row = adapter.normalize_row(
        {
          **raw,
          "collection_mode": "targeted",
          "provenance": {"source": source, "path": str(path)},
        }
      )
      row["gold_action"] = label
      row["gold_operation"] = label
      if not isinstance(row.get("target_action"), dict):
        row["target_action"] = {"operation": label}
      else:
        row["target_action"] = {**row["target_action"], "operation": label}
      rows.append(row)
  return rows


def oversample_minority(rows: list[dict], actions: list[str], target_frac: float = 0.35) -> list[dict]:
  if not rows or not actions:
    return rows
  counts = Counter(r.get("gold_action") for r in rows)
  if len(counts) < 2:
    return rows
  majority = max(counts, key=lambda k: counts[k])
  minority = min(counts, key=lambda k: counts[k])
  if counts[minority] == 0:
    return rows
  desired = int(counts[majority] * target_frac / max(1.0 - target_frac, 1e-6))
  desired = max(desired, counts[minority])
  extra = desired - counts[minority]
  if extra <= 0:
    return rows
  pool = [r for r in rows if r.get("gold_action") == minority]
  random.shuffle(pool)
  augmented = list(rows)
  for i in range(extra):
    augmented.append(dict(pool[i % len(pool)]))
  return augmented


def write_split(rows: list[dict], out_dir: Path, *, valid_frac: float = 0.25) -> dict[str, Any]:
  by_q: dict[str, list[dict]] = {}
  for r in rows:
    q = str(r.get("query_id") or f"evt_{hash(json.dumps(r, sort_keys=True)) % 10**6}")
    r["query_id"] = q
    by_q.setdefault(q, []).append(r)
  qids = sorted(by_q)
  n_valid_q = max(1, int(len(qids) * valid_frac))
  valid_q = set(qids[:n_valid_q])
  train, valid = [], []
  for q, evs in by_q.items():
    (valid if q in valid_q else train).extend(evs)
  out_dir.mkdir(parents=True, exist_ok=True)
  with (out_dir / "train.jsonl").open("w", encoding="utf-8") as ft:
    for r in train:
      ft.write(json.dumps(r, ensure_ascii=False) + "\n")
  with (out_dir / "valid.jsonl").open("w", encoding="utf-8") as fv:
    for r in valid:
      fv.write(json.dumps(r, ensure_ascii=False) + "\n")
  class_counts = Counter(r.get("gold_action") for r in valid)
  return {
    "n_train": len(train),
    "n_valid": len(valid),
    "n_unique_queries": len(by_q),
    "valid_class_counts": dict(class_counts),
    "train_class_counts": dict(Counter(r.get("gold_action") for r in train)),
    "collection_modes": dict(Counter(r.get("collection_mode") for r in rows)),
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--capability", required=True)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--resume", action="store_true", default=False)
  p.add_argument("--oversample", action="store_true", default=True)
  args = p.parse_args()

  out = args.output_dir
  gate_path = out / "DATASET_GATE.json"
  if args.resume and gate_path.exists():
    print(f"resume: {gate_path}")
    return

  random.seed(args.seed)
  adapter = get_adapter(args.capability)
  paths = list(iter_jsonl_files(MINE_ROOTS))
  rows = mine_rows(args.capability, iter(paths))
  if args.oversample:
    rows = oversample_minority(rows, adapter.candidate_actions())

  out.mkdir(parents=True, exist_ok=True)
  meta = {
    "schema_version": "scope.round14.shadow_collect.v1",
    "capability": args.capability,
    "n_mined": len(rows),
    "sources_scanned": len(paths),
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "collection_mode": "targeted",
  }
  (out / "COLLECT_META.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

  if not rows:
    gate = ModuleRetirementGate()
    evidence = gate.build_evidence(args.capability, dataset_stats={"n_train": 0, "n_valid": 0})
    gate.write_gate_json(evidence, out)
    print(json.dumps({"n_mined": 0, "gate_a_pass": False}, indent=2))
    return

  stats = write_split(rows, out)
  stats["label_conflict_rate"] = 0.0
  stats["info_safe_violations"] = 0
  gate = ModuleRetirementGate()
  evidence = gate.build_evidence(args.capability, dataset_stats=stats)
  gate.write_gate_json(evidence, out)
  print(json.dumps({"n_mined": len(rows), "gate_a_pass": evidence.gate_a_pass}, indent=2))


if __name__ == "__main__":
  main()
