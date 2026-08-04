#!/usr/bin/env python3
"""Contract gate evaluation for Round 7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round7.common import OUT, load_jsonl, write_json


def evaluate_trace_integrity(trace_dir: Path) -> dict:
    trace_path = trace_dir / "live_dup_decision_trace.jsonl"
    admission_path = trace_dir / "dup_admission_events.jsonl"
    traces = load_jsonl(trace_path)
    admissions = load_jsonl(admission_path)
    event_ids = [t.get("event_id") for t in traces]
    unique = len(set(event_ids)) == len(event_ids)
    silent_fallback = sum(1 for t in traces if t.get("fallback_used") and not t.get("fallback_reason"))
    return {
        "n_trace_events": len(traces),
        "n_admission_events": len(admissions),
        "trace_admission_match": len(traces) == len(admissions),
        "event_id_unique": unique,
        "silent_fallback": silent_fallback,
        "gate_a_pass": len(traces) == len(admissions) and unique and silent_fallback == 0,
    }


def _find_comparison_summary(run_dir: Path) -> Path:
    name = run_dir.name
    short = name.removesuffix("_tau0") if name.endswith("_tau0") else name
    candidates = [
        OUT / "contract_trace/comparisons" / f"{short}_rerun" / "comparison_summary.json",
        OUT / "contract_trace/comparisons" / f"{name}_rerun" / "comparison_summary.json",
        OUT / "contract_trace/comparisons" / name / "comparison_summary.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = sorted(
        (OUT / "contract_trace/comparisons").glob(f"**/{name}*/comparison_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return found[0] if found else candidates[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()

    gate_a = evaluate_trace_integrity(args.run_dir)
    comp_path = _find_comparison_summary(args.run_dir)
    gate_b = {"gate_b_pass": False}
    if comp_path.exists():
        gate_b = json.loads(comp_path.read_text(encoding="utf-8"))

    result = {
        "run_dir": str(args.run_dir),
        "gate_a": gate_a,
        "gate_b": gate_b,
        "contract_gate_pass": gate_a.get("gate_a_pass") and gate_b.get("gate_b_pass"),
    }
    write_json(args.run_dir / "contract_gate.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["contract_gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
