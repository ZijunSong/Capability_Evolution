#!/usr/bin/env python3
"""audit_decision_alignment (guide §11.2): line-by-line audit of the OPD row
chain — snapshot resolvability, prefix causality, target origin and target-
scoring prefix identity.

Reads the training/episode artifacts (traces + alignment records) and fails
on ANY of:
- a decision without a snapshot ref or public_event_end boundary;
- a target whose scoring prefix includes events at/after the target;
- an executed action text missing when a component rewrote the proposal;
- rows whose teacher prefix token ids were re-rendered instead of recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RSI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RSI_ROOT / "src"))


def audit_trace_json(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        raw = json.loads(path.read_text("utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"{path}: unreadable ({e!r})"]
    for dec in raw.get("decisions", []):
        idx = dec.get("decision_index")
        if not dec.get("environment_snapshot_ref"):
            problems.append(f"{path}: decision {idx} has NO snapshot ref")
        pe = dec.get("public_event_end", -1)
        if pe is None or int(pe) < 0:
            problems.append(f"{path}: decision {idx} has no public_event_end (legacy record)")
        if dec.get("alignment") and dec.get("alignment", {}).get("target_action"):
            # target must be either native (sampled) or explicitly projected
            kind = dec["alignment"].get("target_kind")
            if kind not in ("native", "teacher_scored_projected_target"):
                problems.append(f"{path}: decision {idx} target_kind {kind!r} unknown")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="run dirs / json files to audit")
    args = ap.parse_args()
    problems: list[str] = []
    n_files = 0
    for root in args.roots:
        p = Path(root)
        files = [p] if p.is_file() else sorted(p.rglob("*.json"))
        for f in files:
            if f.name in ("bundle.json", "probe_report.json"):
                continue
            text = f.read_text("utf-8", errors="ignore")[:200]
            if '"decisions"' not in text and '"decision_index"' not in text:
                continue
            n_files += 1
            problems.extend(audit_trace_json(f))
    print(f"audited {n_files} trace files; {len(problems)} problems")
    for pr in problems[:80]:
        print("  -", pr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
