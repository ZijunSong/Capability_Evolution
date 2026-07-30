#!/usr/bin/env python3
"""Build MICRO_OVERFIT_MATRIX.md from B3 results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round5/micro_overfit"
OBJECTIVES = ["O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7"]
SIZES = ["D2", "D8", "D32", "D128"]


def main() -> None:
    lines = [
        "# Round 5 Micro-Overfit Matrix",
        "",
        "| Objective | D2 | D8 | D32 | D128 | All Pass |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for obj in OBJECTIVES:
        summ_path = OUT / obj / "summary.json"
        if not summ_path.exists():
            lines.append(f"| {obj} | — | — | — | — | — |")
            continue
        summ = json.loads(summ_path.read_text())
        cells = []
        for sz in SIZES:
            rep = summ.get("sizes", {}).get(sz)
            if rep is None:
                cells.append("—")
            else:
                acc = rep.get("post_eval", {}).get("operation_accuracy", 0)
                mark = "✅" if rep.get("passed") else "❌"
                cells.append(f"{mark} {acc:.0%}")
        all_pass = "✅" if summ.get("all_pass") else "❌"
        lines.append(f"| {obj} | {' | '.join(cells)} | {all_pass} |")

    (OUT / "MICRO_OVERFIT_MATRIX.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'MICRO_OVERFIT_MATRIX.md'}")


if __name__ == "__main__":
    main()
