#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyopd.methods.scape_component_opd.h1004_post_sweep import gate_from_stats, load_handoff, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stats = read_json(args.stats)
    handoff = load_handoff(args.handoff) if args.handoff else None
    payload = gate_from_stats(args.component, stats, handoff=handoff)
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
