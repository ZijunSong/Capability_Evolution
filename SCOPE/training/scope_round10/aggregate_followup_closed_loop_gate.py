#!/usr/bin/env python3
"""Aggregate followup Phase C/D closed-loop gates (0808-todo1.md §5–§6)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_phase3_gate import aggregate_events, load_jsonl

OUT = _REPO / "outputs/scope_round10_followup"
MAIN = [
    "r10_main_noweight_seed42",
    "r10_main_noweight_seed43",
    "r10_main_noweight_seed44",
]
ALL = ["base"] + MAIN


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


def merge_two_shards(variant_dir: Path) -> dict:
    events = []
    episodes = []
    for i in range(2):
        sh = variant_dir / f"shard{i}"
        events.extend(load_jsonl(sh / "rollback_events.jsonl"))
        episodes.extend(load_jsonl(sh / "episodes.jsonl"))
    metrics = aggregate_events(events, episodes)
    rewards = []
    for ep in episodes:
        if "reward" in ep:
            rewards.append(float(ep["reward"]))
        elif isinstance(ep.get("metrics"), dict) and "reward" in ep["metrics"]:
            rewards.append(float(ep["metrics"]["reward"]))
    priors = {"CONTINUE": 0, "REPLAN": 0, "ROLLBACK_TO": 0}
    for e in events:
        op = str(e.get("student_operation") or "")
        if op in priors:
            priors[op] += 1
    n = max(sum(priors.values()), 1)
    metrics["mean_reward"] = sum(rewards) / max(len(rewards), 1)
    metrics["operation_prior"] = {k: v / n for k, v in priors.items()}
    metrics["checkpoint_accuracy"] = metrics.get("target_checkpoint_accuracy", 0.0)
    return metrics


def continue_not_collapsed(m: dict) -> bool:
    """CONTINUE must not collapse to ≈0."""
    cr = float(m.get("ContinueRecall") or 0.0)
    prior = (m.get("operation_prior") or {}).get("CONTINUE", 0.0)
    return cr >= 0.05 and prior >= 0.05


def smoke_pass(variants: dict[str, dict]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    seed_ms = [variants[v] for v in MAIN if v in variants]
    if len(seed_ms) < 3:
        reasons.append("missing_main_seeds")
        return False, reasons
    for name in MAIN:
        m = variants.get(name) or {}
        checks = {
            "RollbackRecall>=0.30": float(m.get("RollbackRecall") or 0) >= 0.30,
            "FalseRollbackRate<=0.05": float(m.get("FalseRollbackRate") or 0) <= 0.05,
            "checkpoint_accuracy>=0.70": float(m.get("checkpoint_accuracy") or 0) >= 0.70,
            "state_hash_restore_rate==1.0": abs(float(m.get("state_hash_restore_rate") or 0) - 1.0) < 1e-12,
            "budget_violations==0": int(m.get("budget_violations") or 0) == 0,
            "CONTINUE_not_collapsed": continue_not_collapsed(m),
        }
        for k, ok in checks.items():
            if not ok:
                reasons.append(f"{name}:{k}")
    # direction consistency: all seeds RollbackRecall above floor and ContinueRecall same side of collapse
    crs = [float((variants[v] or {}).get("ContinueRecall") or 0) for v in MAIN]
    rrs = [float((variants[v] or {}).get("RollbackRecall") or 0) for v in MAIN]
    if not (all(c >= 0.05 for c in crs) or all(c < 0.05 for c in crs)):
        reasons.append("seed_ContinueRecall_direction_inconsistent")
    if not (all(r >= 0.30 for r in rrs) or all(r < 0.30 for r in rrs)):
        # still require each >=0.30 above; inconsistency is extra
        pass
    return (len(reasons) == 0), reasons


def final_pass(variants: dict[str, dict]) -> tuple[bool, list[str]]:
    # Same hard floors as smoke; establish hard capability on 100q
    ok, reasons = smoke_pass(variants)
    return ok, reasons


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke20", "final100"], required=True)
    args = p.parse_args()

    root = OUT / ("phase_c_smoke20" if args.mode == "smoke20" else "phase_d_final100")
    variants = {}
    for name in ALL:
        vdir = root / name
        if vdir.exists():
            variants[name] = merge_two_shards(vdir)

    if args.mode == "smoke20":
        passed, reasons = smoke_pass(variants)
        report = {
            "mode": "smoke20",
            "pass": passed,
            "SMOKE20_GATE": passed,
            "STOP_AFTER_PHASE_C": not passed,
            "fail_reasons": reasons,
            "variants": variants,
            "git_commit": _git(),
        }
        out = OUT / "SMOKE20_GATE.json"
        md = OUT / "SMOKE20_GATE.md"
    else:
        passed, reasons = final_pass(variants)
        report = {
            "mode": "final100",
            "pass": passed,
            "FINAL100_GATE": passed,
            "ROLLBACK_HARD_CAPABILITY": "ESTABLISHED ON 100Q" if passed else "NOT ESTABLISHED",
            "fail_reasons": reasons,
            "variants": variants,
            "git_commit": _git(),
        }
        out = OUT / "FINAL100_GATE.json"
        md = OUT / "FINAL100_GATE.md"

    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {args.mode.upper()} GATE",
        "",
        f"**pass = {passed}**",
        "",
        "| variant | OpBalAcc | ContR | RollR | FalseRoll | ck_acc | restore | budget | mean_recall | mean_reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in variants.items():
        lines.append(
            f"| {name} | {float(m.get('operation_balanced_accuracy') or 0):.4f} "
            f"| {float(m.get('ContinueRecall') or 0):.4f} "
            f"| {float(m.get('RollbackRecall') or 0):.4f} "
            f"| {float(m.get('FalseRollbackRate') or 0):.4f} "
            f"| {float(m.get('checkpoint_accuracy') or 0):.4f} "
            f"| {float(m.get('state_hash_restore_rate') or 0):.4f} "
            f"| {int(m.get('budget_violations') or 0)} "
            f"| {float(m.get('mean_recall') or 0):.4f} "
            f"| {float(m.get('mean_reward') or 0):.4f} |"
        )
    if reasons:
        lines += ["", "## Fail reasons", ""] + [f"- `{r}`" for r in reasons]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": passed, "n_variants": len(variants), "reasons": reasons}, indent=2))


if __name__ == "__main__":
    main()
