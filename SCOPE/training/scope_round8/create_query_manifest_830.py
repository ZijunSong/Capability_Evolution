#!/usr/bin/env python3
"""Create frozen 830q BrowseComp+ manifest with 4 shards for Round 8."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.opd.browsecomp_queries import load_browsecomp_full_queries


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-queries", type=int, default=830)
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "artifacts/datasets/scope_round8/query_manifest_830.json",
    )
    args = p.parse_args()

    records = load_browsecomp_full_queries(split="all", limit=0, download_if_missing=False)
    qids = sorted([r.query_id for r in records])
    if args.n_queries > 0:
        qids = qids[:args.n_queries]
    assert len(qids) == args.n_queries

    shard_size = len(qids) // args.n_shards
    shards: dict[str, list[str]] = {}
    for i in range(args.n_shards):
        start = i * shard_size
        end = start + shard_size if i < args.n_shards - 1 else len(qids)
        shards[f"shard{i}"] = qids[start:end]

    manifest = {
        "schema_version": "scope.round8.query_manifest.v1",
        "seed": args.seed,
        "n_queries": len(qids),
        "query_ids": qids,
        "shards": shards,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "BrowseComp+",
        "model": "Qwen2.5-7B-Instruct",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output} ({len(qids)} queries, {args.n_shards} shards)")


if __name__ == "__main__":
    main()
