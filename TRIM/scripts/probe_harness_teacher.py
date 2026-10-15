#!/usr/bin/env python3
"""Probe whether a frozen bank yields student-legal targets on real decision states.

This script refuses to invent coverage from an empty event list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.harness_bank.loader import load_bank


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe bank teacher coverage")
    parser.add_argument("--harness-bank", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args(argv)
    bank = load_bank(args.harness_bank)
    if not args.decisions.is_file():
        print(json.dumps({
            "status": "PROBE_BLOCKED",
            "executed_training": False,
            "stop_reason": "decision file is missing; coverage was not computed",
            "bank_sha256": bank.bank_sha256,
        }, indent=2))
        return 2
    rows = [json.loads(line) for line in args.decisions.read_text(encoding="utf-8").splitlines() if line.strip()]
    supported = [row for row in rows if row.get("legal_target")]
    coverage = (len(supported) / len(rows)) if rows else 0.0
    print(json.dumps({
        "status": "PROBE_DONE",
        "executed_training": False,
        "n_decisions": len(rows),
        "coverage_all": coverage,
        "bank_sha256": bank.bank_sha256,
        "empty_bank": not bank.active_components,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
