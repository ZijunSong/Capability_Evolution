#!/usr/bin/env python3
"""Pre/post OPD module marginal value comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.lifecycle.distillability import compute_distillability, compute_module_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre/post OPD module value")
    parser.add_argument("--full-before", type=float, required=True)
    parser.add_argument("--minus-before", type=float, required=True)
    parser.add_argument("--full-after", type=float, required=True)
    parser.add_argument("--minus-after", type=float, required=True)
    parser.add_argument("--output", default="outputs/pre_post_opd.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delta_before = compute_module_delta(args.full_before, args.minus_before)
    delta_after = compute_module_delta(args.full_after, args.minus_after)
    bare_gain = args.minus_after - args.minus_before
    full_gain = args.full_after - args.full_before
    distillability = compute_distillability(delta_before, delta_after)

    result = {
        "delta_before": delta_before,
        "delta_after": delta_after,
        "bare_gain": bare_gain,
        "full_gain": full_gain,
        "distillability": distillability,
        "opd_success_signal": bare_gain > 0 and delta_after < delta_before,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
