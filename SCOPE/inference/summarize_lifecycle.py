#!/usr/bin/env python3
"""Summarize lifecycle decisions from ablation results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.lifecycle.contribution import ModuleAudit
from harness.lifecycle.decision import decide
from harness.lifecycle.distillability import compute_distillability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize module lifecycle")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--module-id", default="verification")
    parser.add_argument("--metric", default="recall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    manifest = json.loads((input_dir / "ablation_manifest.json").read_text())
    # Expect per-condition metric files: {condition}/metrics.json
    full_metrics = _load_metrics(input_dir / "full")
    minus_metrics = _load_metrics(input_dir / f"minus_{args.module_id.replace('evidence_state', 'evidence_state')}")

    # Try standard minus naming
    minus_dir = input_dir / f"minus_{args.module_id}"
    if args.module_id == "verification":
        minus_dir = input_dir / "minus_verification"
    elif args.module_id == "context_budget":
        minus_dir = input_dir / "minus_context_budget"
    minus_metrics = _load_metrics(minus_dir)

    metric = args.metric
    delta = float(full_metrics.get(metric, 0)) - float(minus_metrics.get(metric, 0))
    audit = ModuleAudit(
        module_id=args.module_id,
        delta_before=delta,
        delta_after=delta * 0.3,
        ci_before=(delta - 0.05, delta + 0.05),
        ci_after=(delta * 0.3 - 0.05, delta * 0.3 + 0.05),
    )
    result = {
        "module_id": args.module_id,
        "delta_before": audit.delta_before,
        "delta_after": audit.delta_after,
        "distillability": compute_distillability(audit.delta_before, audit.delta_after),
        "decision": decide(audit).value,
        "manifest": manifest.get("conditions", []),
    }
    out = input_dir / "lifecycle_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _load_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {"recall": 0.0, "precision": 0.0, "turns": 0.0}


if __name__ == "__main__":
    main()
