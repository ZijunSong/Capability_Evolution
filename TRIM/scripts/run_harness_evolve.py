#!/usr/bin/env python3
"""Run bank search without starting TRIM weight training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.evolution.proposer import ProposerClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a harness bank and stop before training")
    parser.add_argument("--proposer-mode", choices=("self", "assisted"), default="self")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model", default="current")
    args = parser.parse_args(argv)
    client = ProposerClient(mode=args.proposer_mode, endpoint=args.endpoint or None, model=args.model)
    try:
        client.propose({"generation": 0})
    except Exception as exc:
        print(json.dumps({
            "status": "SEARCH_NOT_STARTED",
            "executed_training": False,
            "updated": False,
            "stop_reason": str(exc),
        }, indent=2, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
