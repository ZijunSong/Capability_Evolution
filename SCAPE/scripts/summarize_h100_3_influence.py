#!/usr/bin/env python3
"""Summarize H100-3 influence results into a markdown report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=REPO / "outputs" / "h100_3_influence" / "INFLUENCE_BY_COMPONENT.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "outputs" / "h100_3_influence" / "H100_3_INFLUENCE_REPORT.md",
    )
    args = ap.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ranked = sorted(rows, key=lambda r: float(r["normalized_influence"]), reverse=True)
    top = ranked[:4]
    low = ranked[-3:]
    mean = sum(float(r["normalized_influence"]) for r in rows) / len(rows) if rows else 0.0

    lines = [
        "# H100-3 Influence Report",
        "",
        "## Setting",
        f"- input: `{args.input}`",
        f"- components: {len(rows)}",
        f"- mean_normalized_influence: {mean:.6f}",
        "",
        "## Top components",
        "| component | normalized influence | event support |",
        "|---|---:|---:|",
    ]
    for r in top:
        lines.append(
            f"| {r['component']} | {float(r['normalized_influence']):.6f} | {r['event_support']} |"
        )
    lines.extend([
        "",
        "## Lowest components",
        "| component | normalized influence | event support |",
        "|---|---:|---:|",
    ])
    for r in low:
        lines.append(
            f"| {r['component']} | {float(r['normalized_influence']):.6f} | {r['event_support']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "- This report is generated from the offline deterministic scorer.",
        "- Full Harness-1 smoke still requires explicit external-code authorization and a compatible retrieval backend.",
    ])
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
