#!/usr/bin/env python3
"""Barrier 2.1: Query-level live split from frozen base_live."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import (
    DATA,
    R9_DATA,
    class_distribution,
    enrich_from_frozen,
    load_jsonl,
    split_queries,
    write_json,
    write_jsonl,
)

SPLIT_DIR = DATA / "live_split"
BASE_LIVE = R9_DATA / "frozen_replay/base_live.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = load_jsonl(BASE_LIVE)
    by_q: dict[str, list[dict]] = defaultdict(list)
    for row in raw:
        by_q[str(row.get("query_id", ""))].append(row)

    splits = split_queries(list(by_q.keys()), seed=args.seed)
    out_rows: dict[str, list[dict]] = {k: [] for k in splits}
    for split_name, qids in splits.items():
        qset = set(qids)
        for qid in qids:
            for row in by_q.get(qid, []):
                sample = enrich_from_frozen(row, state_source="live")
                if sample:
                    out_rows[split_name].append(sample)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in out_rows.items():
        write_jsonl(SPLIT_DIR / f"{name}.jsonl", rows)

    # Verify each split has both classes
    checks = {}
    for name, rows in out_rows.items():
        dist = class_distribution(rows, binary=True)
        checks[name] = {
            "n_events": len(rows),
            "n_queries": len(splits[name]),
            "distribution": dist,
            "has_continue": dist.get("CONTINUE", 0) > 0,
            "has_rollback": dist.get("ROLLBACK_TO", 0) > 0,
        }

    overlap = set(splits["live_train"]) & set(splits["live_test"])
    report = [
        "# Live Split Report",
        "",
        f"Seed: {args.seed}",
        f"Total queries: {len(by_q)}",
        "",
        "## Split sizes",
        "",
    ]
    for name, c in checks.items():
        report.append(
            f"- **{name}**: {c['n_queries']} queries, {c['n_events']} events, "
            f"dist={c['distribution']}, both_classes={c['has_continue'] and c['has_rollback']}"
        )
    report.append("")
    report.append(f"Query overlap train∩test: {len(overlap)} (must be 0)")

    write_json(SPLIT_DIR / "SPLIT_MANIFEST.json", {"splits": splits, "checks": checks})
    (SPLIT_DIR / "SPLIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Split complete: { {k: len(v) for k, v in out_rows.items()} }")


if __name__ == "__main__":
    main()
