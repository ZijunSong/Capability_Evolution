#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from easyopd.methods.scape_component_opd.h1004_post_sweep import run_reload_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_reload_audit(model_path=args.model, adapter_root=args.adapter_root, output_dir=args.output_dir or None)
    print(result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
