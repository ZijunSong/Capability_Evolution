"""Harness rollout helpers: resume, manifest, retrieval preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


def load_completed_query_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["query_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def check_retrieval_backend(
    retrieval: Literal["chroma", "bm25"] = "bm25",
    *,
    bm25_index_path: str | Path | None = None,
    smoke: bool = False,
) -> str:
    """Fail fast when retrieval prerequisites are missing."""
    if retrieval == "bm25":
        if smoke:
            from harness.retrieval.memory_backend import InMemoryBm25Backend

            InMemoryBm25Backend()
            return "memory://smoke"

        from harness.retrieval.bm25_backend import (
            BrowseCompBm25Backend,
            resolve_bm25_index_path,
        )

        resolved = resolve_bm25_index_path(bm25_index_path)
        BrowseCompBm25Backend(resolved)
        return str(resolved)

    from harness.config import get_config

    config = get_config()
    if config.chroma_api_key.get_secret_value() in {"", "EXAMPLE"}:
        raise RuntimeError(
            "CHROMA_API_KEY is not configured. Edit BiSHOP/.env with real "
            "Chroma Cloud credentials, or use --retrieval bm25 for local BM25."
        )
    if config.chroma_database in {"", "EXAMPLE"}:
        raise RuntimeError(
            "CHROMA_DATABASE is not configured. Edit BiSHOP/.env, or use "
            "--retrieval bm25 for local BM25."
        )
    client = config.get_chroma_client()
    client.heartbeat()
    return "chroma"


def save_harness_manifest(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    jsonl_name: str = "harness_rollouts.jsonl",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / jsonl_name
    total = 0
    if jsonl_path.exists():
        total = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
    payload = {
        "mode": "harness",
        "n_episodes": total,
        "output": str(jsonl_path),
        **manifest,
    }
    path = output_dir / "harness_rollout_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
