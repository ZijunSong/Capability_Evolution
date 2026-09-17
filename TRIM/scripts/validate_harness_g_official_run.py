#!/usr/bin/env python3
"""Official Harness-G result-dir validator. Prints OFFICIAL_RUN_VALID or INVALID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.harness_g_official import validate_official_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Harness-G official run directory")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    result = validate_official_run(args.run_dir)
    print(result["status"])
    for item in result.get("failures") or []:
        print(f"[FAIL] {item}")
    if not result["ok"]:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
