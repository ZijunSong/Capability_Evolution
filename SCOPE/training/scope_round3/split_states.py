#!/usr/bin/env python3
"""Split existing decision_states.jsonl into 8 query shards for bilateral labeling."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, nargs="+", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--n-shards", type=int, default=8)
    args = p.parse_args()

    rows: list[dict] = []
    for sp in args.states:
        with sp.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[str(r.get("query_id", ""))].append(r)
    qids = sorted(by_q.keys())
    n = args.n_shards
    size = len(qids) // n

    for i in range(n):
        start = i * size
        end = start + size if i < n - 1 else len(qids)
        shard_qids = set(qids[start:end])
        out = args.output_root / f"shard{i}"
        out.mkdir(parents=True, exist_ok=True)
        shard_rows = [r for q in shard_qids for r in by_q[q]]
        with (out / "decision_states.jsonl").open("w", encoding="utf-8") as f:
            for r in shard_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "summary.json").write_text(
            json.dumps({"n_queries": len(shard_qids), "n_states": len(shard_rows)}, indent=2)
        )
        print(f"shard{i}: {len(shard_qids)} queries, {len(shard_rows)} states")


if __name__ == "__main__":
    main()
