#!/usr/bin/env python3
"""Generate ROUND5_REPORT.md from pipeline artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round5"


def main() -> None:
    lines = ["# Round 5 Report", ""]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    lines.append(f"- commit: `{commit}`")
    lines.append(f"- stage: `{(OUT / 'PIPELINE_STAGE').read_text().strip() if (OUT / 'PIPELINE_STAGE').exists() else 'n/a'}`")

    for flag in ("B1_PASS", "B2_PASS", "B4_PASS", "ROUND5_COMPLETE"):
        f = OUT / flag
        lines.append(f"- {flag}: `{f.read_text().strip() if f.exists() else 'n/a'}`")

    b3 = OUT / "B3_PASSED_OBJECTIVES"
    if b3.exists():
        lines.append(f"- B3 passed: `{b3.read_text().strip()}`")

    matrix = OUT / "micro_overfit/MICRO_OVERFIT_MATRIX.md"
    if matrix.exists():
        lines.append("")
        lines.append("## Micro-overfit")
        lines.append(matrix.read_text())

    gate = OUT / "b4_full/B4_GATE.json"
    if gate.exists():
        lines.append("")
        lines.append("## B4 Gate")
        lines.append("```json")
        lines.append(gate.read_text())
        lines.append("```")

    (OUT / "ROUND5_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'ROUND5_REPORT.md'}")


if __name__ == "__main__":
    main()
