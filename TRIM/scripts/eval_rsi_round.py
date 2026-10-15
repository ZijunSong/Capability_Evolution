#!/usr/bin/env python3
"""Read a frozen four-grid manifest and score promotion. Does not launch eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.rsi.promotion import evaluate_promotion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one frozen RSI four-grid")
    parser.add_argument("--grid", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.grid.read_text(encoding="utf-8"))
    report = evaluate_promotion(cells=payload["cells"])
    report["executed_training"] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["promotable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
