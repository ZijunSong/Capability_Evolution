"""Per-rank shards. Empty shards must publish DONE.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.training.sft_ablation.contracts import SFTBatchRow, fingerprint
from trim.training.sft_ablation.io_util import atomic_write_json, atomic_write_text, read_json


def shard_dir(out: Path, *, run_id: str, phase: str, round_id: int, rank: int) -> Path:
    return Path(out) / "shards" / run_id / phase / f"{int(round_id):06d}" / str(int(rank))


def write_shard(
    directory: Path,
    rows: Sequence[SFTBatchRow],
    *,
    identity: Mapping[str, Any],
    round_id: int,
    policy_digest: str,
    query_ids: Sequence[str],
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = [row.to_dict() for row in rows]
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload)
    atomic_write_text(directory / "rows.jsonl", text)
    atomic_write_json(
        directory / "DONE.json",
        {
            "experiment": identity["experiment"],
            "identity": fingerprint(dict(identity)),
            "round_id": int(round_id),
            "policy_digest": str(policy_digest),
            "query_ids": list(query_ids),
            "n_rows": len(payload),
            "checksum": fingerprint(payload),
        },
    )


def merge_shards(
    directories: Sequence[Path],
    *,
    identity: Mapping[str, Any],
    round_id: int,
    policy_digest: str,
    expected_query_ids: Sequence[str],
) -> list[SFTBatchRow]:
    seen_queries: list[str] = []
    rows: list[SFTBatchRow] = []
    want_identity = fingerprint(dict(identity))
    for directory in directories:
        done_path = Path(directory) / "DONE.json"
        if not done_path.is_file():
            raise ValueError(f"missing shard DONE.json: {directory}")
        done = read_json(done_path)
        if done.get("experiment") != identity["experiment"] or done.get("identity") != want_identity:
            raise ValueError(f"shard identity mismatch: {directory}")
        if int(done.get("round_id", -1)) != int(round_id):
            raise ValueError(f"shard round mismatch: {directory}")
        if str(done.get("policy_digest")) != str(policy_digest):
            raise ValueError(f"stale shard policy digest: {directory}")
        raw_rows = []
        text = (Path(directory) / "rows.jsonl").read_text(encoding="utf-8") if (Path(directory) / "rows.jsonl").is_file() else ""
        if text.strip():
            raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if fingerprint(raw_rows) != done.get("checksum"):
            raise ValueError(f"shard checksum mismatch: {directory}")
        if len(raw_rows) != int(done.get("n_rows", -1)):
            raise ValueError(f"shard row count mismatch: {directory}")
        seen_queries.extend(str(item) for item in done.get("query_ids") or [])
        rows.extend(SFTBatchRow.from_dict(item) for item in raw_rows)
    if sorted(seen_queries) != sorted(str(item) for item in expected_query_ids):
        raise ValueError("shard query coverage mismatch")
    if len(seen_queries) != len(set(seen_queries)):
        raise ValueError("duplicate query ids across shards")
    ids = [row.row_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row_id")
    rows.sort(key=lambda row: (row.query_id, row.episode_id, row.decision_id, row.row_id))
    return rows
