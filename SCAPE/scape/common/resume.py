"""Resume helpers: never overwrite completed shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResumeError(RuntimeError):
    pass


def completed_query_ids(predictions_path: Path, *, id_key: str = "query_id") -> set[str]:
    if not predictions_path.exists():
        return set()
    ids: set[str] = set()
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get(id_key)
            if qid is None:
                raise ResumeError(f"prediction row missing {id_key}")
            if str(qid) in ids:
                raise ResumeError(f"duplicate query_id in predictions: {qid}")
            ids.add(str(qid))
    return ids


def plan_resume(
    all_query_ids: list[str],
    predictions_path: Path,
    *,
    id_key: str = "query_id",
) -> dict[str, Any]:
    done = completed_query_ids(predictions_path, id_key=id_key)
    remaining = [q for q in all_query_ids if q not in done]
    missing_expected = sorted(done - set(all_query_ids))
    if missing_expected:
        raise ResumeError(
            f"predictions contain query_ids not in manifest: {missing_expected[:5]}"
        )
    return {
        "n_total": len(all_query_ids),
        "n_done": len(done),
        "n_remaining": len(remaining),
        "done": sorted(done),
        "remaining": remaining,
    }


def assert_unique_merge(rows: list[dict[str, Any]], *, id_key: str = "query_id") -> None:
    seen: set[str] = set()
    for row in rows:
        qid = str(row[id_key])
        if qid in seen:
            raise ResumeError(f"duplicate query_id during merge: {qid}")
        seen.add(qid)
