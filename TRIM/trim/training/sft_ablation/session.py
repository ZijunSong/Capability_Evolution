"""Shared run directory, contract seal, and metric lines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.contracts import require_same_identity
from trim.training.sft_ablation.identity import build_contract
from trim.training.sft_ablation.io_util import atomic_write_json, read_json
from trim.training.sft_ablation.query_pool import (
    corpus_digest,
    load_train_queries,
    retrieval_digest,
    write_query_manifest,
)
from trim.training.dist_runtime import barrier


def prepare_directory(args: Any, dist: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = Path(args.out)
    if int(dist.rank) == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier()
    queries, info = load_train_queries(args)
    if int(dist.rank) == 0:
        write_query_manifest(out, queries, info)
    barrier()
    return queries, info


def seal_contract(args: Any, dist: Any, *, query_info: dict[str, Any], archive_digest: str) -> dict[str, Any]:
    contract = build_contract(
        args,
        query_order_digest=str(query_info["order_digest"]),
        corpus_digest=corpus_digest(args),
        retrieval_index_digest=retrieval_digest(args),
        teacher_archive_digest=archive_digest,
        nnodes=int(getattr(dist, "world_size", 1) or 1),
        rollout_replicas=int(os.environ.get("TRIM_ROLLOUT_REPLICAS") or 1),
    )
    path = Path(args.out) / "CONTRACT.json"
    if int(dist.rank) == 0:
        if path.is_file():
            saved = read_json(path)["immutable"]
            require_same_identity(saved, contract["immutable"])
        else:
            atomic_write_json(path, contract)
    barrier()
    saved = read_json(path)
    require_same_identity(saved["immutable"], contract["immutable"])
    return saved


def append_metrics(out: Path, record: dict[str, Any]) -> None:
    import json

    payload = {
        "rl_enabled": False,
        "gap_enabled": False,
        "gate_enabled": False,
        "reward_used_for_optimization": False,
        **record,
    }
    path = Path(out) / "METRICS.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
