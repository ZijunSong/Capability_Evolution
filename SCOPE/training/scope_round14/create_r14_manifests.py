#!/usr/bin/env python3
"""Round14 fresh/query-disjoint manifest builder (seed=1414)."""

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


def load_exclude_set(*paths: Path) -> set[str]:
  out: set[str] = set()
  for p in paths:
    if p.exists():
      out.update(load_query_ids(p))
  return out


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
    "schema_version": "scope.round14.query_manifest.v1",
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
  p.add_argument("--seed", type=int, default=1414)
  p.add_argument(
    "--source-manifest",
    type=Path,
    default=_REPO / "artifacts/datasets/scope_round8/query_manifest_830.json",
  )
  p.add_argument(
    "--exclude-audit",
    type=Path,
    default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
  )
  p.add_argument(
    "--exclude-r13-dir",
    type=Path,
    default=_REPO / "artifacts/datasets/scope_round13/manifests",
  )
  p.add_argument(
    "--out-dir",
    type=Path,
    default=_REPO / "artifacts/datasets/scope_round14/manifests",
  )
  p.add_argument(
    "--freeze-json",
    type=Path,
    default=_REPO / "outputs/scope_round14/RUN_MANIFEST.json",
  )
  args = p.parse_args()

  all_qids = load_query_ids(args.source_manifest)
  exclude = load_exclude_set(args.exclude_audit)
  if args.exclude_r13_dir.exists():
    for mf in sorted(args.exclude_r13_dir.glob("R13_*.json")):
      exclude.update(load_query_ids(mf))

  pool = [q for q in all_qids if q not in exclude]
  pool_sorted = sorted(pool, key=lambda q: stable_rank(q, args.seed))

  need = [
    ("R14_FRESH100", 100, 1),  # single shard: full-100 retirement rollouts
    ("R14_SMOKE20", 20, 1),
    ("R14_TRAIN_POOL", 300, 4),
    ("R14_HOLD_830", len(all_qids), 8),
  ]

  cursor = 0
  splits: dict[str, list[str]] = {}
  for name, n, _ in need[:-1]:
    splits[name] = pool_sorted[cursor : cursor + n]
    cursor += n
  splits["R14_HOLD_830"] = list(all_qids)

  names = [n for n, _, _ in need[:-1]]
  for i, a in enumerate(names):
    for b in names[i + 1 :]:
      inter = set(splits[a]) & set(splits[b])
      if inter:
        raise RuntimeError(f"overlap {a}∩{b}={len(inter)}")
    if set(splits[a]) & exclude:
      raise RuntimeError(f"{a} intersects exclude set")

  args.out_dir.mkdir(parents=True, exist_ok=True)
  freeze: dict = {
    "schema_version": "scope.round14.run_manifest.v1",
    "seed": args.seed,
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_manifest": str(args.source_manifest),
    "exclude_audit": str(args.exclude_audit),
    "exclude_r13_dir": str(args.exclude_r13_dir),
    "n_pool_after_exclude": len(pool_sorted),
    "n_excluded": len(exclude),
    "manifests": {},
  }

  shard_map = {
    "R14_FRESH100": 2,
    "R14_SMOKE20": 1,
    "R14_TRAIN_POOL": 4,
    "R14_HOLD_830": 8,
  }
  for name, n, n_shards in need:
    qids = splits[name]
    meta = {
      "dataset": "BrowseComp+",
      "excluded_audit100": True,
      "excluded_r13": args.exclude_r13_dir.exists(),
      "requested_n": n,
      "primary_eval": name == "R14_FRESH100",
    }
    info = write_manifest(
      args.out_dir / f"{name}.json",
      name=name,
      seed=args.seed,
      qids=qids,
      n_shards=shard_map.get(name, n_shards),
      meta=meta,
    )
    freeze["manifests"][name] = info

  # Secondary references (not primary eval)
  r13_refs = {}
  r13_dir = _REPO / "artifacts/datasets/scope_round13/manifests"
  for ref in ("R13_FINAL100", "R13_SMOKE20"):
    src = r13_dir / f"{ref}.json"
    if src.exists():
      r13_refs[ref] = {"path": str(src), "sha256": sha256_file(src)}
  freeze["secondary_manifests"] = r13_refs

  args.freeze_json.parent.mkdir(parents=True, exist_ok=True)
  args.freeze_json.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
  main()
