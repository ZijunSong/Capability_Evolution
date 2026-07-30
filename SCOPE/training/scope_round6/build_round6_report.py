#!/usr/bin/env python3
"""Generate ROUND6_REPORT.md and ROUND6_GATE.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.common import OUT, git_commit, SEEDS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT)
    args = p.parse_args()
    out = args.output_dir

    lines = [
        "# Round 6 Report",
        "",
        f"- commit: `{git_commit()}`",
        "",
    ]

    gate_path = out / "phase_b/ROOT_CAUSE_GATE.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        lines.append("## Root Cause Gate")
        for k in ("H_RUNTIME", "H_CALIB", "H_SHIFT", "H_FEEDBACK"):
            lines.append(f"- {k}: `{gate.get(k)}`")
        lines.append("")

    csv_path = out / "phase_b/CROSS_SCORE_MATRIX.csv"
    if csv_path.exists():
        lines.append("## Cross-Score Matrix")
        lines.append("")
        lines.append(csv_path.read_text())
        lines.append("")

    closed = out / "closed_loop"
    holdout_reports = []
    if closed.exists():
        for d in sorted(closed.rglob("aggregated_metrics.json")):
            holdout_reports.append((d.parent.name, json.loads(d.read_text())))

    gate_result = {
        "ROUND6_CLOSED_LOOP_POSITIVE": False,
        "RECOMMEND_830": False,
        "gates": {},
    }

    if holdout_reports:
        lines.append("## Closed-Loop Holdout")
        for name, rep in holdout_reports:
            d = rep.get("direct_behavior", {})
            lines.append(
                f"- {name}: DupRejectRecall={d.get('DupRejectRecall', 0):.3f} "
                f"FSR={d.get('FalseSkipRate', 0):.3f} reward={rep.get('mean_reward', 0):.3f}"
            )

    # Final go/no-go placeholder — updated after Phase D
    lines.append("")
    lines.append("## Recommendation")
    rec = gate_result["RECOMMEND_830"]
    lines.append(f"`RECOMMEND_830={'true' if rec else 'false'}`")

    (out / "ROUND6_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "ROUND6_GATE.json").write_text(json.dumps(gate_result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'ROUND6_REPORT.md'}")


if __name__ == "__main__":
    main()
