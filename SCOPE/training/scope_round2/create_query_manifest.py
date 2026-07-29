#!/usr/bin/env python3
"""Create frozen query manifest for Round 2 100q audit."""

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
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument(
        "--source",
        type=Path,
        default=_REPO / "artifacts/datasets/e0_audit_100q/query_ids.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    p.add_argument("--n-shards", type=int, default=4)
    args = p.parse_args()

    data = json.loads(args.source.read_text(encoding="utf-8"))
    qids = [str(x) for x in data.get("query_ids", data)][: args.n_queries]
    assert len(qids) == args.n_queries, f"expected {args.n_queries} queries, got {len(qids)}"

    shard_size = args.n_queries // args.n_shards
    shards: dict[str, list[str]] = {}
    for i in range(args.n_shards):
        start = i * shard_size
        end = start + shard_size if i < args.n_shards - 1 else args.n_queries
        shards[f"shard{i}"] = qids[start:end]

    manifest = {
        "schema_version": "scope.round2.query_manifest.v1",
        "seed": args.seed,
        "n_queries": args.n_queries,
        "query_ids": qids,
        "shards": shards,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "BrowseComp+",
        "model": "Qwen2.5-7B-Instruct",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} ({args.n_queries} queries, {args.n_shards} shards)")


if __name__ == "__main__":
    main()
