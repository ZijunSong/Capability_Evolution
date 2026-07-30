#!/usr/bin/env python3
"""Build Round 4 overfit128 dataset: 64 KEEP + 64 SKIP, query-deduped."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl, write_json
from harness.capability.dup_operation import DupOperation


def _query_key(sample: dict) -> str:
    ds = sample.get("decision_state") or {}
    return str(ds.get("task_id") or ds.get("episode_id") or sample.get("episode_id") or sample.get("sample_id"))


def _operation(sample: dict) -> str | None:
    ct = compact_target_from_sample(sample)
    if ct is None:
        return None
    return ct.operation.value


def sample_balanced_overfit(
    train_rows: list[dict],
    *,
    n_keep: int = 64,
    n_skip: int = 64,
    seed: int = 42,
) -> list[dict]:
    keep_pool: dict[str, list[dict]] = defaultdict(list)
    skip_pool: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        op = _operation(row)
        qk = _query_key(row)
        if op == DupOperation.KEEP_EVIDENCE.value:
            keep_pool[qk].append(row)
        elif op == DupOperation.SKIP_DUPLICATE.value:
            skip_pool[qk].append(row)

    rng = random.Random(seed)
    keep_queries = list(keep_pool.keys())
    skip_queries = list(skip_pool.keys())
    rng.shuffle(keep_queries)
    rng.shuffle(skip_queries)

    selected: list[dict] = []
    for q in keep_queries:
        if len(selected) >= n_keep:
            break
        selected.append(rng.choice(keep_pool[q]))
    keep_count = len(selected)

    for q in skip_queries:
        if len(selected) >= n_keep + n_skip:
            break
        selected.append(rng.choice(skip_pool[q]))
    skip_count = len(selected) - keep_count

    rng.shuffle(selected)
    return selected, keep_count, skip_count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round3/train.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round4_overfit128",
    )
    p.add_argument("--n-keep", type=int, default=64)
    p.add_argument("--n-skip", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_rows = load_jsonl(args.train)
    selected, n_keep, n_skip = sample_balanced_overfit(
        train_rows, n_keep=args.n_keep, n_skip=args.n_skip, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train.jsonl"
    with train_out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "n_total": len(selected),
        "n_keep": n_keep,
        "n_skip": n_skip,
        "n_unique_queries": len({_query_key(r) for r in selected}),
        "seed": args.seed,
        "source": str(args.train),
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
