#!/usr/bin/env python3
"""Create fixed 20q smoke manifest from 100q manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--input", type=Path, default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    queries = sorted(data.get("queries", data if isinstance(data, list) else []), key=lambda q: str(q.get("query_id", q)))
    subset = queries[: args.n]
    out = {"queries": subset, "n_queries": len(subset), "source": str(args.input)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(subset)} queries -> {args.output}")


if __name__ == "__main__":
    main()
