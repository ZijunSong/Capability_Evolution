#!/usr/bin/env python3
"""Reaggregate Round 8 Phase 3 with fixed metrics and diff vs original."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9 import aggregate_phase3_gate as agg


def _metric_keys() -> list[str]:
    return [
        "operation_balanced_accuracy",
        "ContinueRecall",
        "RollbackRecall",
        "RollbackPrecision",
        "FalseRollbackRate",
        "target_checkpoint_accuracy",
        "state_hash_restore_rate",
        "budget_violations",
        "fallback_count",
        "invalid_checkpoint_rate",
    ]


def build_diff(original: dict, reagg: dict) -> dict:
    diff: dict = {"variants": {}, "summary": {"aggregation_bug_changed": [], "low_accuracy_still_holds": []}}
    orig_vars = original.get("variants", {})
    new_vars = reagg.get("variants", {})
    for variant in sorted(set(orig_vars) | set(new_vars)):
        ov = orig_vars.get(variant, {})
        nv = new_vars.get(variant, {})
        row = {}
        for key in _metric_keys():
            old = ov.get(key)
            new = nv.get(key)
            if old is None and new is None:
                continue
            delta = None if old is None or new is None else new - old
            row[key] = {"original": old, "reaggregated": new, "delta": delta}
            if delta is not None and abs(delta) > 1e-6:
                if key in ("state_hash_restore_rate", "budget_violations"):
                    diff["summary"]["aggregation_bug_changed"].append(f"{variant}:{key}")
                elif key in ("operation_balanced_accuracy", "target_checkpoint_accuracy", "ContinueRecall"):
                    if (new or 0) < 0.15:
                        diff["summary"]["low_accuracy_still_holds"].append(f"{variant}:{key}")
        diff["variants"][variant] = row
    diff["summary"]["main_seeds_pass_original"] = original.get("main_seeds_pass")
    diff["summary"]["main_seeds_pass_reaggregated"] = reagg.get("main_seeds_pass")
    return diff


def write_md(diff: dict, path: Path) -> None:
    lines = ["# Round 8 Phase 3 metric diff (original vs reaggregated)", ""]
    lines.append("## Summary")
    lines.append(f"- aggregation_bug_changed: {diff['summary']['aggregation_bug_changed']}")
    lines.append(f"- low_accuracy_still_holds: {diff['summary']['low_accuracy_still_holds']}")
    lines.append(
        f"- main_seeds_pass: {diff['summary']['main_seeds_pass_original']} -> "
        f"{diff['summary']['main_seeds_pass_reaggregated']}"
    )
    lines.append("")
    for variant, metrics in diff.get("variants", {}).items():
        lines.append(f"## {variant}")
        for key, row in metrics.items():
            lines.append(
                f"- {key}: {row.get('original')} -> {row.get('reaggregated')} "
                f"(delta={row.get('delta')})"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, default=_REPO / "outputs/scope_round8/HARD_CAPABILITY_GATE_PHASE3.json")
    p.add_argument("--out-dir", type=Path, default=_REPO / "outputs/scope_round9/reaggregate_round8")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "HARD_CAPABILITY_GATE_PHASE3_REAGG.json"
    sys.argv = [
        "aggregate_phase3_gate.py",
        "--output",
        str(out_json),
    ]
    agg.main()

    original = json.loads(args.original.read_text(encoding="utf-8")) if args.original.exists() else {}
    reagg = json.loads(out_json.read_text(encoding="utf-8"))
    diff = build_diff(original, reagg)
    (args.out_dir / "metric_diff_vs_original.json").write_text(
        json.dumps(diff, indent=2) + "\n", encoding="utf-8"
    )
    write_md(diff, args.out_dir / "metric_diff_vs_original.md")
    print(f"Wrote diff to {args.out_dir}")


if __name__ == "__main__":
    main()
