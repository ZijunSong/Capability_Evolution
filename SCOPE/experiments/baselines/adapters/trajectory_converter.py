"""Convert SCOPE trajectories to external baseline schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def scope_episode_to_seed(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": episode.get("query_id"),
        "question": episode.get("question") or episode.get("query"),
        "messages": episode.get("messages") or episode.get("trajectory") or [],
        "reward": episode.get("reward"),
        "answer": episode.get("final_answer") or episode.get("answer"),
    }


def scope_episode_to_opid(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": episode.get("query_id"),
        "query": episode.get("question") or episode.get("query"),
        "steps": episode.get("steps") or episode.get("events") or [],
        "score": episode.get("reward") or episode.get("recall"),
    }


def scope_episode_to_sdar(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": episode.get("query_id"),
        "prompt": episode.get("question") or episode.get("query"),
        "trajectory": episode.get("trajectory") or episode.get("messages") or [],
        "outcome": episode.get("reward"),
    }


CONVERTERS = {
    "SEED": scope_episode_to_seed,
    "OPID": scope_episode_to_opid,
    "SDAR": scope_episode_to_sdar,
}


def convert_jsonl(src: Path, dst: Path, *, baseline: str) -> int:
    if baseline not in CONVERTERS:
        raise ValueError(f"unknown baseline: {baseline}")
    fn = CONVERTERS[baseline]
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(fn(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    return n
