#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyopd.methods.scape_component_opd.h1004_post_sweep import discover_component_handoffs, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = discover_component_handoffs(args.output_root)
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
