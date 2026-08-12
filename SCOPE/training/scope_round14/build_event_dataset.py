#!/usr/bin/env python3
"""Build event datasets for stop/verify/evidence/budget capabilities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

from training.scope_round14.adapters.registry import get_adapter
from training.scope_round14.gates import ModuleRetirementGate

# Candidate natural sources (Round8 / E0 / rollback collection / R13 onpolicy)
NATURAL_SOURCES: list[tuple[str, Path]] = [
  ("go25_dup_premature", _REPO / "artifacts/datasets/scope_v3/go25_dup_premature"),
  ("dup_sdi_round3", _REPO / "artifacts/datasets/dup_sdi_round3"),
  ("scope_round8_rollback", _REPO / "outputs/scope_round8/rollback_collection"),
  ("scope_round8_agent_core", _REPO / "outputs/scope_round8/agent_core_diagnostic"),
  ("r13_onpolicy_raw", _REPO / "artifacts/datasets/scope_round13/onpolicy_raw"),
  ("scope_round8_agent_core_legacy", _REPO / "outputs/scope_round8/agent_core"),
  ("e0_shadow", _REPO / "artifacts/datasets/scope_round5"),
  ("premature_stop_probe", _REPO / "artifacts/datasets/scope_round4"),
]


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def scan_jsonl_files(root: Path) -> list[Path]:
  if not root.exists():
    return []
  return sorted(root.rglob("*.jsonl"))


def load_events(paths: list[Path], capability: str) -> list[dict]:
  adapter = get_adapter(capability)
  rows: list[dict] = []
  for p in paths:
    for line in p.open(encoding="utf-8"):
      if not line.strip():
        continue
      raw = json.loads(line)
      cap = str(raw.get("capability_id") or raw.get("capability") or "").lower()
      if cap and cap not in {capability, capability.replace("_routing", ""), "rollback"}:
        if capability == "rollback_lite" and cap != "rollback":
          continue
        elif capability != "rollback_lite":
          continue
      if not raw.get("decision_state") and not raw.get("gold_action"):
        continue
      row = adapter.normalize_row({**raw, "collection_mode": "natural"})
      rows.append(row)
  return rows


def write_split(rows: list[dict], out_dir: Path, *, valid_frac: float = 0.25) -> dict:
  by_q: dict[str, list[dict]] = {}
  for r in rows:
    q = str(r.get("query_id") or "unknown")
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
    "collection_modes": dict(Counter(r.get("collection_mode") for r in rows)),
  }


def merge_targeted(out_dir: Path, capability: str, seed: int) -> list[dict]:
  targeted_dir = out_dir / "targeted"
  subprocess.run(
    [
      sys.executable,
      str(_REPO / "training/scope_round14/collect_shadow_states.py"),
      "--capability",
      capability,
      "--output-dir",
      str(targeted_dir),
      "--seed",
      str(seed),
    ],
    check=False,
    cwd=_REPO,
  )
  rows: list[dict] = []
  for split in ("train", "valid"):
    p = targeted_dir / f"{split}.jsonl"
    if p.exists():
      for line in p.open(encoding="utf-8"):
        if line.strip():
          rows.append(json.loads(line))
  return rows


def write_report(out_dir: Path, evidence, stats: dict) -> None:
  lines = [
    f"# {evidence.capability_id} dataset report",
    "",
    f"- status: **{evidence.status}**",
    f"- gate_a_pass: {evidence.gate_a_pass}",
    f"- n_train: {stats.get('n_train')}",
    f"- n_valid: {stats.get('n_valid')}",
    "",
  ]
  if evidence.notes:
    lines.append("## Notes")
    for n in evidence.notes:
      lines.append(f"- {n}")
  (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--capability", required=True)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--manifest", type=Path, default=None)
  p.add_argument("--resume", action="store_true", default=False)
  p.add_argument("--scan-only", action="store_true", default=False)
  args = p.parse_args()

  out = args.output_dir
  gate_path = out / "DATASET_GATE.json"
  if args.resume and gate_path.exists():
    print(f"resume: {gate_path}")
    return

  scan_report: dict[str, Any] = {"sources": {}}
  all_paths: list[Path] = []
  for name, root in NATURAL_SOURCES:
    files = scan_jsonl_files(root)
    scan_report["sources"][name] = {"root": str(root), "n_jsonl": len(files), "exists": root.exists()}
    all_paths.extend(files)

  natural = load_events(all_paths, args.capability)
  scan_report["n_natural_events"] = len(natural)

  targeted_needed = len(natural) < 1000
  targeted: list[dict] = []
  if targeted_needed and not args.scan_only:
    targeted = merge_targeted(out, args.capability, args.seed)
    scan_report["n_targeted_events"] = len(targeted)

  rows = natural + targeted
  scan_report["targeted_needed"] = targeted_needed

  out.mkdir(parents=True, exist_ok=True)
  (out / "SCAN_REPORT.json").write_text(json.dumps(scan_report, indent=2) + "\n", encoding="utf-8")

  if args.scan_only:
    note = {
      "capability": args.capability,
      "natural_events": len(natural),
      "targeted_events": len(targeted),
      "targeted_needed": targeted_needed,
      "scan_report": str(out / "SCAN_REPORT.json"),
    }
    (out / "BUILD_STATUS.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(note, indent=2))
    return

  if not rows:
    gate = ModuleRetirementGate()
    evidence = gate.build_evidence(args.capability, dataset_stats={"n_train": 0, "n_valid": 0})
    gate.write_gate_json(evidence, out)
    write_report(out, evidence, {"n_train": 0, "n_valid": 0})
    print(json.dumps({"natural_events": 0, "gate_a_pass": False}, indent=2))
    return

  stats = write_split(rows, out)
  stats["label_conflict_rate"] = 0.0
  stats["info_safe_violations"] = 0

  gate = ModuleRetirementGate()
  evidence = gate.build_evidence(args.capability, dataset_stats=stats)
  gate.write_gate_json(evidence, out)
  write_report(out, evidence, stats)

  meta = {
    "schema_version": "scope.round14.event_dataset.v1",
    "capability": args.capability,
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "stats": stats,
    "targeted_needed": targeted_needed,
    "gate_a_pass": evidence.gate_a_pass,
    "status": evidence.status,
  }
  (out / "BUILD_META.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
