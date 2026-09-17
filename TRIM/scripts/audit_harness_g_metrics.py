#!/usr/bin/env python3
"""Independently recompute official Harness-G metrics from PER_QUERY.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.harness_g_contract import audit_trace_metrics


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit official Harness-G metrics from raw traces")
    parser.add_argument("--per-query", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_rows(args.per_query)
    failures: list[str] = []
    for row in rows:
        failures.extend(audit_trace_metrics(row))
    n = len(rows)
    if failures:
        print(f"FAIL {len(failures)} mismatches over {n} rows")
        for item in failures[:20]:
            print(" ", item)
        return 1
    print(f"PASS {n}/{n} rows metric audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
