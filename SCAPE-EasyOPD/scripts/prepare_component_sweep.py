#!/usr/bin/env python3
"""Freeze query-disjoint manifests for the SCAPE component sweep.

This utility only prepares query contracts. It never fabricates rollouts or
component events; event-state collection remains gated on the real runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path(
    "/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"
)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "SCAPE" / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(query_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).hexdigest()


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", ""))
            query = row.get("query")
            if not query_id or not isinstance(query, str) or not query.strip():
                raise ValueError(f"invalid query row at {path}:{line_number}")
            if query_id in seen:
                raise ValueError(f"duplicate query_id={query_id}")
            seen.add(query_id)
            rows.append({"query_id": query_id, "query": query})
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=8191)
    parser.add_argument("--train-max", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=128)
    parser.add_argument("--test", type=int, default=256)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source query manifest does not exist: {args.source}")
    if args.train_max < 1000 or args.train_max > 2000:
        raise SystemExit("--train-max must be within the protocol range [1000, 2000]")

    rows = load_queries(args.source)
    ranked = sorted(rows, key=lambda row: stable_rank(row["query_id"], args.seed))
    required = args.train_max + args.dev + args.test
    if len(ranked) < required:
        # Use all real queries, preserving the protocol shortfall explicitly.
        train_end = max(0, len(ranked) - args.dev - args.test)
        train_end = min(train_end, args.train_max)
    else:
        train_end = args.train_max
    train = ranked[:train_end]
    dev = ranked[train_end : train_end + args.dev]
    test = ranked[train_end + args.dev : train_end + args.dev + args.test]

    sets = {"train": train, "dev": dev, "test": test}
    ids = {name: {row["query_id"] for row in subset} for name, subset in sets.items()}
    if ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"]:
        raise AssertionError("query split overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(args.source)
    for name, subset in sets.items():
        write_json(
            args.output_dir / f"COMPONENT_SWEEP_{name.upper() if name != 'train' else 'TRAIN_POOL'}.json",
            {
                "schema_version": "scape_component_sweep_query_manifest_v1",
                "split": "TRAIN_POOL" if name == "train" else name.upper(),
                "source": str(args.source),
                "source_sha256": source_sha,
                "seed": args.seed,
                "query_ids": [row["query_id"] for row in subset],
                "query_count": len(subset),
                "query_disjoint": True,
                "contains_answers": False,
                "notes": "Query contract only; answers/gold docs are intentionally excluded from runtime inputs.",
            },
        )

    status = "READY_FOR_REAL_COLLECTION" if len(train) >= 1000 and len(dev) == args.dev and len(test) == args.test else "QUERY_POOL_INSUFFICIENT"
    write_json(
        args.output_dir / "COMPONENT_SWEEP_QUERY_CONTRACT.json",
        {
            "schema_version": "scape_component_sweep_query_contract_v1",
            "status": status,
            "source": str(args.source),
            "source_sha256": source_sha,
            "seed": args.seed,
            "counts": {name: len(subset) for name, subset in sets.items()},
            "query_disjoint": True,
            "real_runtime_required": True,
            "event_collection_status": "BLOCKED_RUNTIME_NOT_READY",
            "canonical_student_base": None,
            "reason": "EasyOPD handoff does not provide CANONICAL_STUDENT_BASE and harness/scape runtime is unavailable in the approved environment.",
        },
    )
    print(json.dumps({"status": status, "counts": {name: len(subset) for name, subset in sets.items()}, "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
