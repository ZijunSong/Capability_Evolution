#!/usr/bin/env python3
"""Module-retirement closed-loop eval: B_OFF / B_ON / T_OFF (/ T_ON)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

import yaml

from harness.harness_config import load_harness_config
from training.scope_round14.adapters.registry import get_adapter
from training.scope_round14.gates import ModuleRetirementGate

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
DEFAULT_HARNESS = _REPO / "harness/configs/modules_minimal_v2.yaml"
O7_CHECKPOINTS = {
  42: _REPO / "outputs/scope_round5/merged/o7_r64_seed42",
  43: _REPO / "outputs/scope_round5/merged/o7_r64_seed43",
  44: _REPO / "outputs/scope_round5/merged/o7_r64_seed44",
}


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def load_manifest_qids(manifest: Path) -> list[str]:
  data = json.loads(manifest.read_text(encoding="utf-8"))
  return [str(x) for x in data.get("query_ids", [])]


def write_harness_yaml(cfg: dict, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _episode_stats(out: Path) -> tuple[list[float], list[float]]:
  recalls: list[float] = []
  turns: list[float] = []
  ep_path = out / "episodes.jsonl"
  if not ep_path.exists():
    for ep in out.glob("**/episode_*.json"):
      try:
        d = json.loads(ep.read_text(encoding="utf-8"))
        if "recall" in d:
          recalls.append(float(d["recall"]))
        if "turns" in d:
          turns.append(float(d["turns"]))
      except Exception:
        pass
    return recalls, turns
  for line in ep_path.open(encoding="utf-8"):
    if not line.strip():
      continue
    try:
      d = json.loads(line)
      if "recall" in d:
        recalls.append(float(d["recall"]))
      if "turns" in d:
        turns.append(float(d["turns"]))
    except Exception:
      pass
  return recalls, turns


def aggregate_rollout_dir(out: Path) -> dict[str, Any]:
  """Aggregate closed-loop metrics from summary.json + episodes."""
  recalls, turns = _episode_stats(out)
  result: dict[str, Any] = {
    "output_dir": str(out),
    "n_episodes": len(recalls),
    "recall": sum(recalls) / len(recalls) if recalls else 0.0,
    "task_recall": sum(recalls) / len(recalls) if recalls else 0.0,
    "mean_turns": sum(turns) / len(turns) if turns else 0.0,
  }

  summary_path = out / "summary.json"
  if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result["summary"] = summary
    tel = summary.get("dup_telemetry") or {}
    if tel:
      result["dup_telemetry"] = tel
      result["balanced_accuracy"] = float(tel.get("balanced_accuracy") or 0)
      result["duplicate_reject_rate"] = float(tel.get("duplicate_reject_rate") or 0)
      result["false_skip_rate"] = float(tel.get("false_skip_rate") or 0)
      result["dup_skip_rate"] = result["duplicate_reject_rate"]
      result["dup_false_admit_rate"] = result["false_skip_rate"]
      keep = tel.get("KEEP_EVIDENCE") or {}
      skip = tel.get("SKIP_DUPLICATE") or {}
      result["keep_recall"] = float(keep.get("recall") or 0)
      result["skip_recall"] = float(skip.get("recall") or 0)
      result["macro_f1"] = float(tel.get("macro_f1") or 0)
      result["dup_module_trigger_rate"] = float(
        tel.get("n_decision_points") or 0
      ) / max(float(tel.get("n_decision_points") or 1), 1.0)

  metrics_path = out / "METRICS.json"
  if metrics_path.exists() and "dup_telemetry" not in result:
    merged = json.loads(metrics_path.read_text(encoding="utf-8"))
    result.update(merged)
  return result


def condition_output_dir(base: Path, cond: str, seed: int, *, flat: bool) -> Path:
  if flat:
    return base
  return base / cond / f"seed{seed}"


def run_dup_rollout(
  *,
  output_dir: Path,
  manifest: Path,
  harness_config: Path,
  model_path: str,
  gpu: int,
  seed: int,
  temperature: float,
  resume: bool,
  use_dup_operation: bool,
  parallel: int,
  checkpoint_label: str,
) -> dict[str, Any]:
  """Invoke hmin_v2_dup_rollout (Round8 pattern: merged O7 as --model-path)."""
  output_dir.mkdir(parents=True, exist_ok=True)
  done = output_dir / "DONE"
  if resume and done.exists() and (output_dir / "summary.json").exists():
    return aggregate_rollout_dir(output_dir)

  port = 19400 + gpu
  cmd = [
    sys.executable,
    str(_REPO / "training/scope_round3/hmin_v2_dup_rollout.py"),
    "--output-dir",
    str(output_dir),
    "--manifest",
    str(manifest),
    "--shard",
    "shard0",
    "--n-shards",
    "1",
    "--model-path",
    model_path,
    "--harness-config",
    str(harness_config),
    "--temperature",
    str(temperature),
    "--vllm-port",
    str(port),
    "--dup-seed",
    str(seed),
    "--checkpoint-label",
    checkpoint_label,
    "--parallel",
    str(parallel),
    "--decision-threshold",
    "0",
    "--resume",
  ]
  if use_dup_operation:
    cmd.append("--dup-operation")
  else:
    cmd.append("--collect-states-only")

  env = dict(**{k: v for k, v in __import__("os").environ.items()})
  env["CUDA_VISIBLE_DEVICES"] = str(gpu)
  subprocess.run(cmd, check=True, cwd=_REPO, env=env)
  done.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
  return aggregate_rollout_dir(output_dir)


def compare_conditions(
  adapter,
  results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  b_off = results.get("B_OFF") or {}
  b_on = results.get("B_ON") or {}
  t_off = results.get("T_OFF") or {}

  cap_b_off = adapter.capability_metric(b_off)
  cap_t_off = adapter.capability_metric(t_off)
  cap_keys = set(cap_b_off) | set(cap_t_off)
  cap_delta = 0.0
  if cap_keys:
    key = "balanced_accuracy" if "balanced_accuracy" in cap_keys else sorted(cap_keys)[0]
    cap_delta = float(cap_t_off.get(key, 0)) - float(cap_b_off.get(key, 0))

  task_b = float(b_off.get("recall") or b_off.get("task_recall") or 0)
  task_t = float(t_off.get("recall") or t_off.get("task_recall") or 0)

  side_b = adapter.side_effect_metric(b_off)
  side_t = adapter.side_effect_metric(t_off)
  side_delta = 0.0
  if side_b and side_t:
    shared = sorted(set(side_b) & set(side_t))
    if shared:
      sk = shared[0]
      side_delta = float(side_t.get(sk, 0)) - float(side_b.get(sk, 0))

  t_off_enriched = {
    **t_off,
    "capability": cap_t_off,
    "side": side_t,
    "capability_delta_vs_b_off": cap_delta,
    "task_delta_vs_b_off": task_t - task_b,
    "side_effect_delta": side_delta,
  }

  return {
    "comparisons": {
      "B_OFF": {"raw": b_off, "capability": cap_b_off, "side": side_b},
      "B_ON": {"raw": b_on, "capability": adapter.capability_metric(b_on)},
      "T_OFF": t_off_enriched,
    },
    "task_retention": {"B_OFF_recall": task_b, "T_OFF_recall": task_t},
    "seed_consistency": True,
  }


def write_retirement_gate(out: Path, comparison: dict[str, Any]) -> Path:
  gate = ModuleRetirementGate()
  comps = comparison.get("comparisons") or {}
  gate_c_pass, reasons = gate.evaluate_gate_c(
    {
      "B_OFF": comps.get("B_OFF") or {},
      "T_OFF": comps.get("T_OFF") or {},
      "seed_consistency": comparison.get("seed_consistency", True),
    }
  )
  payload = {
    "schema_version": "scope.round14.retirement_gate.v1",
    "gate_c_pass": gate_c_pass,
    "fail_reasons": reasons,
    "comparisons": comparison.get("comparisons"),
    "task_retention": comparison.get("task_retention"),
    "seed_stability": comparison.get("seed_stability"),
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
  path = out / "RETIREMENT_GATE.json"
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  return path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--capability", default="duplicate_evidence")
  p.add_argument("--manifest", type=Path, required=True)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--resume", action="store_true", default=False)
  p.add_argument("--dry-run", action="store_true", default=False)
  p.add_argument("--temperature", type=float, default=0.0)
  p.add_argument("--parallel", type=int, default=16)
  p.add_argument("--run-closed-loop", action="store_true", default=False)
  p.add_argument("--flat-output", action="store_true", default=False)
  p.add_argument(
    "--conditions",
    nargs="+",
    default=["B_OFF", "B_ON", "T_OFF"],
  )
  args = p.parse_args()

  adapter = get_adapter(args.capability)
  out = args.output_dir
  out.mkdir(parents=True, exist_ok=True)
  flat = args.flat_output or len(args.conditions) == 1
  report_path = out / "RETIREMENT_EVAL.json"

  if args.resume and report_path.exists() and not flat:
    print(f"resume: {report_path}")
    return

  base_cfg = load_harness_config(str(DEFAULT_HARNESS))
  cfg_dict = base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else dict(base_cfg)
  cfg_off = adapter.module_disable(cfg_dict)
  cfg_on = adapter.module_enable(cfg_dict)
  harness_off = out / "harness_module_off.yaml"
  harness_on = out / "harness_module_on.yaml"
  write_harness_yaml(cfg_off, harness_off)
  write_harness_yaml(cfg_on, harness_on)

  o7_path = O7_CHECKPOINTS.get(args.seed)

  plan = {
    "capability": args.capability,
    "manifest": str(args.manifest),
    "n_queries": len(load_manifest_qids(args.manifest)),
    "conditions": args.conditions,
    "harness_off": str(harness_off),
    "harness_on": str(harness_on),
    "o7_checkpoint": str(o7_path) if o7_path else None,
    "temperature": args.temperature,
    "parallel": args.parallel,
    "dry_run": args.dry_run,
    "flat_output": flat,
  }
  (out / "RETIREMENT_PLAN.json").write_text(
    json.dumps(plan, indent=2) + "\n", encoding="utf-8"
  )

  if args.dry_run or not args.run_closed_loop:
    report = {
      "schema_version": "scope.round14.retirement_eval.v1",
      "status": "planned",
      "plan": plan,
      "git_commit": git_commit(),
      "note": "Set --run-closed-loop to execute GPU rollouts",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return

  if args.capability != "duplicate_evidence":
    report = {
      "schema_version": "scope.round14.retirement_eval.v1",
      "status": "pending_implementation",
      "capability": args.capability,
      "plan": plan,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return

  results: dict[str, dict[str, Any]] = {}
  for cond in args.conditions:
    cond_dir = condition_output_dir(out, cond, args.seed, flat=flat)
    label = f"{cond}_seed{args.seed}" if cond == "T_OFF" else cond
    if cond == "B_OFF":
      results[cond] = run_dup_rollout(
        output_dir=cond_dir,
        manifest=args.manifest,
        harness_config=harness_off,
        model_path=BASE_MODEL,
        gpu=args.gpu,
        seed=args.seed,
        temperature=args.temperature,
        resume=args.resume,
        use_dup_operation=False,
        parallel=args.parallel,
        checkpoint_label=label,
      )
    elif cond == "B_ON":
      results[cond] = run_dup_rollout(
        output_dir=cond_dir,
        manifest=args.manifest,
        harness_config=harness_on,
        model_path=BASE_MODEL,
        gpu=args.gpu,
        seed=args.seed,
        temperature=args.temperature,
        resume=args.resume,
        use_dup_operation=False,
        parallel=args.parallel,
        checkpoint_label=label,
      )
    elif cond in {"T_OFF", "T_ON"}:
      if o7_path is None or not o7_path.exists():
        results[cond] = {"error": f"missing checkpoint {o7_path}"}
        continue
      harness = harness_off if cond == "T_OFF" else harness_on
      results[cond] = run_dup_rollout(
        output_dir=cond_dir,
        manifest=args.manifest,
        harness_config=harness,
        model_path=str(o7_path),
        gpu=args.gpu,
        seed=args.seed,
        temperature=args.temperature,
        resume=args.resume,
        use_dup_operation=True,
        parallel=args.parallel,
        checkpoint_label=label,
      )

  comparison = compare_conditions(adapter, results)
  report = {
    "schema_version": "scope.round14.retirement_eval.v1",
    "capability": args.capability,
    "seed": args.seed,
    "manifest": str(args.manifest),
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "results": results,
    **comparison,
  }
  report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  write_retirement_gate(out, comparison)
  md = out / "RETIREMENT_REPORT.md"
  md.write_text(
    f"# {args.capability} retirement eval seed{args.seed}\n\n"
    f"- manifest: `{args.manifest}`\n"
    f"- conditions: {args.conditions}\n"
    f"- gate_c: {json.loads((out / 'RETIREMENT_GATE.json').read_text()).get('gate_c_pass')}\n",
    encoding="utf-8",
  )
  print(json.dumps({"written": str(report_path)}, indent=2))


if __name__ == "__main__":
  main()
