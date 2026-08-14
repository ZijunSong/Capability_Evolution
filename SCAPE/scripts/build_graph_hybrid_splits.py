#!/usr/bin/env python3
"""Build graph-hybrid V2/V3 query-disjoint splits for SCAPE-0813-Next-H20 Phase B."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.graph_hybrid import build_graph_hybrid_splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs/0813_next_h20/graph_hybrid/data",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    meta = build_graph_hybrid_splits(out_dir=args.out_dir, seed=args.seed)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
