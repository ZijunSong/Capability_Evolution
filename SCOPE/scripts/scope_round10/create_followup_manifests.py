#!/usr/bin/env python3
"""Build frozen smoke20 / final100 manifests with 2-way matched sharding for followup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
SRC = _REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json"


def _split(qids: list[str], n_shards: int) -> dict[str, list[str]]:
    shards: dict[str, list[str]] = {}
    n = len(qids)
    base = n // n_shards
    rem = n % n_shards
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        shards[f"shard{i}"] = [str(x) for x in qids[start : start + size]]
        start += size
    return shards


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    all_ids = [str(x) for x in src["query_ids"]]
    assert len(all_ids) == 100, len(all_ids)

    smoke_ids = all_ids[:20]
    smoke = {
        "schema_version": "scope.round10.followup.smoke20.v1",
        "source": str(SRC),
        "n_queries": len(smoke_ids),
        "query_ids": smoke_ids,
        "shards": _split(smoke_ids, 2),
        "n_shards": 2,
        "note": "First 20 of frozen 100q; matched 2-way sharding for 8-GPU Base/seed42/43/44",
    }
    final = {
        "schema_version": "scope.round10.followup.final100.v1",
        "source": str(SRC),
        "n_queries": len(all_ids),
        "query_ids": all_ids,
        "shards": _split(all_ids, 2),
        "n_shards": 2,
        "note": "Frozen 100q remapped to 2 shards (50+50) for matched 8-GPU layout",
        "original_shards_4way": src.get("shards"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "smoke20.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "final100.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"smoke20": len(smoke_ids), "final100": len(all_ids), "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
