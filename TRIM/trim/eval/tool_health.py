"""Aggregate worker tool-health counters for upstream_api eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_COUNTER_KEYS = (
    "read_success",
    "read_unknown_id",
    "grep_success",
    "grep_no_results",
    "grep_invalid_regex",
    "grep_timeout",
    "verify_requests",
    "verify_valid_verdict",
    "verify_empty_content",
    "verify_parse_failures",
    "verify_length_truncated",
    "verify_length_retries",
)


def build_tool_health_payload(
    capability_log: Mapping[str, Any],
    *,
    worker_rank: int | None = None,
    phase: str = "final",
) -> dict[str, Any]:
    counters = {key: int(capability_log.get(key) or 0) for key in _COUNTER_KEYS}
    payload: dict[str, Any] = {
        "phase": phase,
        "capability_log": dict(capability_log),
        "counters": counters,
    }
    if worker_rank is not None:
        payload["worker_rank"] = int(worker_rank)
    verify_calls = capability_log.get("verify_calls")
    if isinstance(verify_calls, list):
        payload["verify_calls"] = list(verify_calls)
    corpus_validation = capability_log.get("corpus_validation")
    if isinstance(corpus_validation, dict):
        payload["corpus_validation"] = dict(corpus_validation)
    return payload


def write_tool_health(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def merge_tool_health(paths: Sequence[Path]) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    merged_counters = {key: 0 for key in _COUNTER_KEYS}
    merged_log: dict[str, Any] = {}
    verify_calls: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        shards.append(payload)
        counters = payload.get("counters") or {}
        for key in _COUNTER_KEYS:
            merged_counters[key] += int(counters.get(key) or 0)
        log = payload.get("capability_log")
        if isinstance(log, dict):
            for key, value in log.items():
                if key in _COUNTER_KEYS or key == "verify_calls":
                    continue
                merged_log.setdefault(key, value)
        calls = payload.get("verify_calls") or (payload.get("capability_log") or {}).get("verify_calls")
        if isinstance(calls, list):
            verify_calls.extend(calls)
    merged_log.update(merged_counters)
    if verify_calls:
        merged_log["verify_calls"] = verify_calls
    return {
        "phase": "merged_final",
        "n_shards": len(shards),
        "shards": shards,
        "counters": merged_counters,
        "capability_log": merged_log,
        "verify_calls": verify_calls,
    }
