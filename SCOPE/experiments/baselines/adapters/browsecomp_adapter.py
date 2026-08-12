"""Convert SCOPE BrowseComp+ manifests into baseline-friendly query lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # SCOPE Round2/8 style: {query_ids: [...], n_queries, ...}
        if "query_ids" in data and isinstance(data["query_ids"], list):
            return [
                {
                    "query_id": str(qid),
                    "question": None,  # resolved at runtime from BrowseComp+ corpus
                    "dataset": data.get("dataset"),
                    "manifest_schema": data.get("schema_version"),
                }
                for qid in data["query_ids"]
            ]
        for key in ("queries", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # id -> row map (skip non-dict top-level metadata)
        dict_items = {k: v for k, v in data.items() if isinstance(v, dict)}
        if dict_items and all(isinstance(v, dict) for v in dict_items.values()):
            rows = []
            for qid, row in dict_items.items():
                r = dict(row)
                r.setdefault("query_id", qid)
                rows.append(r)
            return rows
    raise ValueError(f"unsupported manifest schema: {path}")


def to_baseline_queries(rows: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    for r in rows[: limit or len(rows)]:
        qid = r.get("query_id") or r.get("id") or r.get("qid")
        if qid is None:
            raise ValueError(f"manifest row missing query_id: {r}")
        question = r.get("question") or r.get("query") or r.get("prompt")
        if question is None:
            # SCOPE manifests often store only IDs; question text is loaded by runtime.
            question = f"[BrowseComp+ unresolved question text for query_id={qid}]"
        out.append(
            {
                "query_id": str(qid),
                "question": question,
                "answer": r.get("answer") or r.get("gold_answer"),
                "metadata": {
                    k: v
                    for k, v in r.items()
                    if k not in {"query_id", "id", "qid", "question", "query", "prompt"}
                },
            }
        )
    return out


def write_baseline_queries(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
