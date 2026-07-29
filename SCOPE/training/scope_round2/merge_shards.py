#!/usr/bin/env python3
"""Merge Round 2 rollout shards and validate query coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_shards(shard_dirs: list[Path], out_dir: Path, manifest_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = set(str(x) for x in manifest["query_ids"])

    all_episodes: list[dict] = []
    all_states: list[dict] = []
    for shard in shard_dirs:
        all_episodes.extend(load_jsonl(shard / "episodes.jsonl"))
        all_states.extend(load_jsonl(shard / "decision_states.jsonl"))

    seen: set[str] = set()
    merged_eps: list[dict] = []
    for ep in all_episodes:
        qid = str(ep.get("query_id", ""))
        if qid and qid not in seen:
            seen.add(qid)
            merged_eps.append(ep)

    missing = expected - seen
    dup = len(all_episodes) - len(merged_eps)
    report = {
        "n_expected": len(expected),
        "n_merged": len(merged_eps),
        "n_missing": len(missing),
        "n_duplicates_removed": dup,
        "missing_query_ids": sorted(missing)[:20],
        "complete": len(missing) == 0 and len(merged_eps) == len(expected),
    }

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in sorted(merged_eps, key=lambda x: str(x.get("query_id"))):
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    with (out_dir / "decision_states.jsonl").open("w", encoding="utf-8") as f:
        for st in all_states:
            f.write(json.dumps(st, ensure_ascii=False) + "\n")
    (out_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not report["complete"]:
        raise SystemExit(f"Merge incomplete: {report}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    args = p.parse_args()
    report = merge_shards(args.shard_dirs, args.output_dir, args.manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
