#!/usr/bin/env python3
"""Barrier 0: freeze Round13 fresh query manifests (seed=1309, audit100-disjoint)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def stable_rank(qid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def load_query_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "query_ids" in data:
        return [str(x) for x in data["query_ids"]]
    raise ValueError(f"unsupported manifest: {path}")


def write_manifest(
    out: Path,
    *,
    name: str,
    seed: int,
    qids: list[str],
    n_shards: int,
    meta: dict,
) -> dict:
    shards: dict[str, list[str]] = {}
    if n_shards <= 1:
        shards["shard0"] = list(qids)
    else:
        for i in range(n_shards):
            shards[f"shard{i}"] = [q for j, q in enumerate(qids) if j % n_shards == i]
    payload = {
        "schema_version": "scope.round13.query_manifest.v1",
        "name": name,
        "seed": seed,
        "n_queries": len(qids),
        "query_ids": qids,
        "shards": shards,
        "n_shards": n_shards,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(out), "sha256": sha256_file(out), "n_queries": len(qids)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1309)
    p.add_argument(
        "--source-manifest",
        type=Path,
        default=_REPO / "artifacts/datasets/scope_round8/query_manifest_830.json",
    )
    p.add_argument(
        "--exclude",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "artifacts/datasets/scope_round13/manifests",
    )
    p.add_argument(
        "--freeze-json",
        type=Path,
        default=_REPO / "outputs/scope_round13/RUN_MANIFEST.json",
    )
    args = p.parse_args()

    all_qids = load_query_ids(args.source_manifest)
    exclude = set(load_query_ids(args.exclude))
    pool = [q for q in all_qids if q not in exclude]
    pool_sorted = sorted(pool, key=lambda q: stable_rank(q, args.seed))

    # Priority: TRAIN200, VALID100, TEST100, SMOKE20, FINAL100 — never shrink FINAL.
    need = [
        ("R13_TRAIN200", 200, 6),
        ("R13_VALID100", 100, 2),
        ("R13_TEST100", 100, 2),
        ("R13_SMOKE20", 20, 1),
        ("R13_FINAL100", 100, 2),
    ]
    total_need = sum(n for _, n, _ in need)
    if len(pool_sorted) < total_need:
        # Shrink TRAIN only.
        deficit = total_need - len(pool_sorted)
        train_n = max(0, 200 - deficit)
        need[0] = ("R13_TRAIN200", train_n, min(6, max(1, train_n // 25)))

    cursor = 0
    splits: dict[str, list[str]] = {}
    for name, n, _ in need:
        splits[name] = pool_sorted[cursor : cursor + n]
        cursor += n

    # Disjointness checks
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = set(splits[a]) & set(splits[b])
            if inter:
                raise RuntimeError(f"overlap {a}∩{b}={len(inter)}")
        if set(splits[a]) & exclude:
            raise RuntimeError(f"{a} intersects audit100")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    freeze: dict = {
        "schema_version": "scope.round13.run_manifest.v1",
        "seed": args.seed,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source_manifest),
        "exclude_manifest": str(args.exclude),
        "n_pool_after_exclude": len(pool_sorted),
        "n_unused_pool": len(pool_sorted) - cursor,
        "manifests": {},
    }
    for name, n, n_shards in need:
        meta = {
            "dataset": "BrowseComp+",
            "excluded_audit100": True,
            "requested_n": n,
        }
        info = write_manifest(
            args.out_dir / f"{name}.json",
            name=name,
            seed=args.seed,
            qids=splits[name],
            n_shards=n_shards,
            meta=meta,
        )
        freeze["manifests"][name] = info
        print(f"Wrote {name}: n={info['n_queries']} sha256={info['sha256'][:16]}...")

    args.freeze_json.parent.mkdir(parents=True, exist_ok=True)
    args.freeze_json.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(f"Froze {args.freeze_json}")


if __name__ == "__main__":
    main()
