#!/usr/bin/env python3
"""Aggregate wave0 Dup anchor conditions into retirement eval + gate + report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

from training.scope_round14.adapters.c0_duplicate_evidence import DuplicateEvidenceAdapter
from training.scope_round14.gates import ModuleRetirementGate
from training.scope_round14.run_module_retirement_eval import aggregate_rollout_dir


def load_condition(root: Path, name: str) -> dict[str, Any] | None:
  cond_dir = root / name
  if not cond_dir.exists():
    return None
  if (cond_dir / "summary.json").exists() or (cond_dir / "DONE").exists():
    return aggregate_rollout_dir(cond_dir)
  return None


def seed_direction_ok(t_off: dict, b_off: dict, adapter: DuplicateEvidenceAdapter) -> bool:
  cap_b = adapter.capability_metric(b_off)
  cap_t = adapter.capability_metric(t_off)
  key = "balanced_accuracy" if "balanced_accuracy" in cap_b else sorted(cap_b)[0]
  return float(cap_t.get(key, 0)) > float(cap_b.get(key, 0))


def build_comparison(
  b_off: dict[str, Any],
  b_on: dict[str, Any],
  t_off_by_seed: dict[int, dict[str, Any]],
) -> dict[str, Any]:
  adapter = DuplicateEvidenceAdapter()
  cap_b_off = adapter.capability_metric(b_off)
  side_b = adapter.side_effect_metric(b_off)
  task_b = float(b_off.get("recall") or b_off.get("task_recall") or 0)

  seed_dirs: dict[str, Any] = {}
  seed_ok: list[bool] = []
  best_seed = None
  best_delta = -1e9
  best_t_off: dict[str, Any] = {}

  for seed, t_off in sorted(t_off_by_seed.items()):
    cap_t = adapter.capability_metric(t_off)
    side_t = adapter.side_effect_metric(t_off)
    task_t = float(t_off.get("recall") or t_off.get("task_recall") or 0)
    cap_keys = set(cap_b_off) | set(cap_t)
    key = "balanced_accuracy" if "balanced_accuracy" in cap_keys else sorted(cap_keys)[0]
    cap_delta = float(cap_t.get(key, 0)) - float(cap_b_off.get(key, 0))
    task_delta = task_t - task_b
    side_delta = 0.0
    shared = sorted(set(side_b) & set(side_t))
    if shared:
      side_delta = float(side_t[shared[0]]) - float(side_b[shared[0]])
    seed_ok.append(cap_delta > 0)
    seed_dirs[f"seed{seed}"] = {
      "raw": t_off,
      "capability": cap_t,
      "capability_delta_vs_b_off": cap_delta,
      "task_delta_vs_b_off": task_delta,
      "side_effect_delta": side_delta,
    }
    if cap_delta > best_delta:
      best_delta = cap_delta
      best_seed = seed
      best_t_off = t_off

  cap_t_best = adapter.capability_metric(best_t_off)
  side_t_best = adapter.side_effect_metric(best_t_off)
  task_t = float(best_t_off.get("recall") or best_t_off.get("task_recall") or 0)
  cap_keys = set(cap_b_off) | set(cap_t_best)
  key = "balanced_accuracy" if "balanced_accuracy" in cap_keys else sorted(cap_keys)[0]
  cap_delta = float(cap_t_best.get(key, 0)) - float(cap_b_off.get(key, 0))
  side_delta = 0.0
  shared = sorted(set(side_b) & set(side_t_best))
  if shared:
    side_delta = float(side_t_best[shared[0]]) - float(side_b[shared[0]])

  t_off_enriched = {
    **best_t_off,
    "best_seed": best_seed,
    "capability": cap_t_best,
    "side": side_t_best,
    "capability_delta_vs_b_off": cap_delta,
    "task_delta_vs_b_off": task_t - task_b,
    "side_effect_delta": side_delta,
    "per_seed": seed_dirs,
  }

  return {
    "comparisons": {
      "B_OFF": {"raw": b_off, "capability": cap_b_off, "side": side_b},
      "B_ON": {"raw": b_on, "capability": adapter.capability_metric(b_on)},
      "T_OFF": t_off_enriched,
    },
    "task_retention": {"B_OFF_recall": task_b, "T_OFF_recall": task_t},
    "seed_consistency": all(seed_ok) if seed_ok else False,
    "seed_stability": {
      "n_seeds": len(t_off_by_seed),
      "direction_positive": seed_ok,
      "best_seed": best_seed,
      "best_capability_delta": cap_delta,
    },
  }


def write_report(out: Path, comparison: dict[str, Any], gate_c_pass: bool, reasons: list[str]) -> None:
  lines = [
    "# Dup retirement anchor (Round14 wave0)",
    "",
    f"- gate_c_pass: **{gate_c_pass}**",
    f"- best_seed: {comparison.get('seed_stability', {}).get('best_seed')}",
    "",
    "## Conditions",
  ]
  for cond in ("B_OFF", "B_ON", "T_OFF"):
    raw = (comparison.get("comparisons") or {}).get(cond, {}).get("raw") or {}
    tel = raw.get("dup_telemetry") or {}
    lines.append(f"### {cond}")
    lines.append(f"- balanced_accuracy: {tel.get('balanced_accuracy', raw.get('balanced_accuracy', 'n/a'))}")
    lines.append(f"- duplicate_reject_rate: {tel.get('duplicate_reject_rate', 'n/a')}")
    lines.append(f"- false_skip_rate: {tel.get('false_skip_rate', 'n/a')}")
    lines.append(f"- episode recall: {raw.get('recall', 'n/a')}")
    lines.append(f"- mean_turns: {raw.get('mean_turns', 'n/a')}")
    lines.append("")
  if reasons:
    lines.append("## Gate C reasons")
    for r in reasons:
      lines.append(f"- {r}")
  (out / "DUP_RETIREMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--anchor-dir",
    type=Path,
    default=_REPO / "outputs/scope_round14/gpu0_dup_anchor",
  )
  p.add_argument("--resume", action="store_true", default=False)
  args = p.parse_args()

  root = args.anchor_dir
  gate_path = root / "DUP_RETIREMENT_GATE.json"
  if args.resume and gate_path.exists():
    print(f"resume: {gate_path}")
    return

  b_off = load_condition(root, "B_OFF")
  b_on = load_condition(root, "B_ON")
  t_off_by_seed: dict[int, dict[str, Any]] = {}
  for seed in (42, 43, 44):
    t = load_condition(root, f"T_OFF_seed{seed}")
    if t:
      t_off_by_seed[seed] = t

  missing = []
  if not b_off:
    missing.append("B_OFF")
  if not b_on:
    missing.append("B_ON")
  if not t_off_by_seed:
    missing.append("T_OFF_seed*")

  if missing:
    payload = {
      "schema_version": "scope.round14.dup_retirement_gate.v1",
      "status": "incomplete",
      "missing": missing,
      "created_at": datetime.now(timezone.utc).isoformat(),
    }
    gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return

  comparison = build_comparison(b_off, b_on or {}, t_off_by_seed)
  gate = ModuleRetirementGate()
  gate_c_pass, reasons = gate.evaluate_gate_c(
    {
      "B_OFF": comparison["comparisons"]["B_OFF"],
      "T_OFF": comparison["comparisons"]["T_OFF"],
      "seed_consistency": comparison.get("seed_consistency", True),
    }
  )

  eval_payload = {
    "schema_version": "scope.round14.retirement_eval.v1",
    "capability": "duplicate_evidence",
    "manifest": "aggregated_wave0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "results": {
      "B_OFF": b_off,
      "B_ON": b_on,
      **{f"T_OFF_seed{s}": m for s, m in t_off_by_seed.items()},
    },
    **comparison,
  }
  (root / "RETIREMENT_EVAL.json").write_text(
    json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )

  gate_payload = {
    "schema_version": "scope.round14.dup_retirement_gate.v1",
    "gate_c_pass": gate_c_pass,
    "fail_reasons": reasons,
    "comparisons": comparison.get("comparisons"),
    "task_retention": comparison.get("task_retention"),
    "seed_stability": comparison.get("seed_stability"),
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
  gate_path.write_text(json.dumps(gate_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  write_report(root, comparison, gate_c_pass, reasons)
  print(json.dumps({"gate_c_pass": gate_c_pass, "best_seed": comparison["seed_stability"]["best_seed"]}, indent=2))


if __name__ == "__main__":
  main()
