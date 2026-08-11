#!/usr/bin/env python3
"""Append a SCAPE stage section to result-record.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scape.common.result_record import append_result_record, format_stage_section

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--setting-json", type=Path, required=True)
    ap.add_argument("--results-json", type=Path, required=True)
    ap.add_argument("--paired-json", type=Path, default=None)
    ap.add_argument("--gate", default="UNRESOLVED")
    ap.add_argument("--decision", default="")
    ap.add_argument("--record", type=Path, default=REPO / "result-record.md")
    args = ap.parse_args()

    setting = json.loads(args.setting_json.read_text(encoding="utf-8"))
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    paired = (
        json.loads(args.paired_json.read_text(encoding="utf-8")) if args.paired_json else None
    )
    section = format_stage_section(
        stage=args.stage,
        setting=setting,
        results=results,
        paired=paired,
        gate=args.gate,
        decision=args.decision,
    )
    append_result_record(args.record, section)
    print(f"appended to {args.record}")


if __name__ == "__main__":
    main()
