#!/usr/bin/env python3
"""Stop calibration 100q audit with quadrant statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState
from harness.capability.stop_calibration import StopQuadrantStats, classify_stop_quadrant
from harness.shadow.verification_shadow import VerificationShadow
from training.scope.schema import Route
from training.scope.routing import route_decision


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_states(states_path: Path) -> dict:
    shadow = VerificationShadow()
    quadrants = StopQuadrantStats()
    routes = Counter()
    n_points = 0

    for row in load_jsonl(states_path):
        ds_raw = row.get("decision_state") or {}
        student_raw = row.get("student_action") or {}
        if not ds_raw or not student_raw:
            continue
        state = DecisionState.from_dict(ds_raw)
        student = CapabilityAction.from_dict(student_raw)
        artifact = shadow.analyze(state, student)
        rec = artifact.recommended_action
        quad = classify_stop_quadrant(student, rec)
        quadrants.record(quad)
        n_points += 1
        routed = route_decision(state, artifact, student)
        routes[routed.route.value] += 1

    report = {
        "n_decision_points": n_points,
        **quadrants.to_dict(),
        "ENDORSE": routes.get("ENDORSE", 0),
        "CORRECT": routes.get("CORRECT", 0),
        "IGNORE": routes.get("IGNORE", 0),
        "bilateral_coverage": quadrants.continue_to_stop > 0 and quadrants.stop_to_continue > 0,
    }
    return report


def render_md(report: dict) -> str:
    lines = [
        "# Stop Calibration 100q Audit (H_min_v2)\n",
        f"- n_decision_points: **{report['n_decision_points']}**",
        f"- bilateral_coverage: **{report.get('bilateral_coverage')}**\n",
        "## Quadrants\n",
        "| Quadrant | Count |",
        "|----------|-------|",
    ]
    for k in ["STOP→STOP", "STOP→CONTINUE", "CONTINUE→STOP", "CONTINUE→CONTINUE"]:
        lines.append(f"| {k} | {report.get(k, 0)} |")
    lines.append(
        f"\n## Routes\n- ENDORSE: {report.get('ENDORSE', 0)}\n"
        f"- CORRECT: {report.get('CORRECT', 0)}\n"
        f"- IGNORE: {report.get('IGNORE', 0)}\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", nargs="+", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()

    combined = StopQuadrantStats()
    routes = Counter()
    n_total = 0
    for sp in args.states:
        r = audit_states(sp)
        n_total += r["n_decision_points"]
        for k in ["STOP→STOP", "STOP→CONTINUE", "CONTINUE→STOP", "CONTINUE→CONTINUE"]:
            combined.record(k)  # wrong - need to merge counts
        routes["ENDORSE"] += r.get("ENDORSE", 0)
        routes["CORRECT"] += r.get("CORRECT", 0)
        routes["IGNORE"] += r.get("IGNORE", 0)

    # Re-audit all states in one pass
    all_rows = []
    for sp in args.states:
        all_rows.extend(load_jsonl(sp))

    quadrants = StopQuadrantStats()
    routes = Counter()
    shadow = VerificationShadow()
    for row in all_rows:
        ds_raw = row.get("decision_state") or {}
        student_raw = row.get("student_action") or {}
        if not ds_raw or not student_raw:
            continue
        state = DecisionState.from_dict(ds_raw)
        student = CapabilityAction.from_dict(student_raw)
        artifact = shadow.analyze(state, student)
        quadrants.record(classify_stop_quadrant(student, artifact.recommended_action))
        routed = route_decision(state, artifact, student)
        routes[routed.route.value] += 1

    report = {
        "n_decision_points": quadrants.n_decision_points,
        **quadrants.to_dict(),
        "ENDORSE": routes.get("ENDORSE", 0),
        "CORRECT": routes.get("CORRECT", 0),
        "IGNORE": routes.get("IGNORE", 0),
        "bilateral_coverage": quadrants.continue_to_stop > 0 and quadrants.stop_to_continue > 0,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_md(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
