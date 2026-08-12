#!/usr/bin/env python3
"""Build H100-4 prestage evidence from completed H100-1/2/3 artifacts.

This does not run any new training or real-model scoring. It consolidates the
already-completed H100-1/2/3 outputs into a stable evidence table for the H20
candidate-selection handoff.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_4_influence_confirm")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    h1001 = REPO / "outputs" / "h100_1_contribution"
    h1001_confirm = REPO / "outputs" / "h100_1_contribution_confirm"
    h1002 = REPO / "outputs" / "h100_2_independent_repl"
    h1003 = REPO / "outputs" / "h100_3_influence_qrel"
    prestage = REPO / "outputs" / "scape_prestage"

    rows = []
    h1001_rows = {r["component"]: r for r in _read_csv(h1001 / "COMPONENT_CONTRIBUTION.csv")}
    h1001c_rows = {r["component"]: r for r in _read_csv(h1001_confirm / "CONTRIBUTION_CONFIRM.csv")}
    h1002_rows = {r["component"]: r for r in _read_csv(h1002 / "LOO_REPLICATION_V2.csv")}
    h1003_rows = {r["component"]: r for r in _read_csv(h1003 / "INFLUENCE_BY_COMPONENT.csv")}

    for component in sorted(h1001_rows):
        row = {
            "component": component,
            "H100-1 CAL200": float(h1001_rows[component].get("delta_harness_reward", 0.0)),
            "H100-1 CONFIRM400": float(h1001c_rows.get(component, {}).get("delta_harness_reward", 0.0)),
            "H100-2 REPL200_V2": float(h1002_rows.get(component, {}).get("delta_harness_reward", 0.0)),
            "H100-3 REAL_INF64": float(h1003_rows.get(component, {}).get("normalized_influence", 0.0)),
            "H100-4 CONFIRM128": "PENDING_REAL_SCORER",
            "placement type": prestage.joinpath("CAPABILITY_PLACEMENT_MAP.json").exists() and "prestage" or "unknown",
        }
        rows.append(row)

    md_lines = [
        "# PRESTAGE_EVIDENCE_TABLE",
        "",
        "| component | H100-1 CAL200 | H100-1 CONFIRM400 | H100-2 REPL200_V2 | H100-3 REAL_INF64 | H100-4 CONFIRM128 | placement type |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['component']} | {row['H100-1 CAL200']:.6f} | {row['H100-1 CONFIRM400']:.6f} | {row['H100-2 REPL200_V2']:.6f} | {row['H100-3 REAL_INF64']:.6f} | {row['H100-4 CONFIRM128']} | {row['placement type']} |"
        )
    (out / "PRESTAGE_EVIDENCE_TABLE.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out / "PRESTAGE_EVIDENCE_TABLE.json").write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
