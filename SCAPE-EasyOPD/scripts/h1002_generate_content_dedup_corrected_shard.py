#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyopd.methods.scape_component_opd.event_collection import generate_real_harness_rollouts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-manifest', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--query-max', type=int, default=2000)
    parser.add_argument('--rollouts-max', type=int, default=4)
    parser.add_argument('--seed-base', type=int, default=20260818)
    parser.add_argument('--query-start', type=int, required=True)
    parser.add_argument('--query-count', type=int, required=True)
    args = parser.parse_args()
    payload = generate_real_harness_rollouts(
        component='content_dedup',
        query_manifest=args.query_manifest,
        output_path=args.output_path,
        query_max=args.query_max,
        rollouts_max=args.rollouts_max,
        seed_base=args.seed_base,
        query_start=args.query_start,
        query_count=args.query_count,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
