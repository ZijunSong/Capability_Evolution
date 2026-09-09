"""Dispatch official Harness-1 eval to isolated API workers (E02)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.upstream_harness1.model_serve import ServedModelIdentity
from trim.upstream_harness1.retrieval import RetrievalConfig
from trim.upstream_harness1.v8d_flags import subprocess_env_for_mask


def run_isolated_api_eval(
    *,
    rows: Sequence[Mapping[str, Any]],
    out: Path,
    harness: str,
    harness_mask: Mapping[str, bool],
    identity: ServedModelIdentity,
    retrieval: RetrievalConfig,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    pool_meta: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Spawn one worker process that imports ultra_core after V8D flags are set."""
    retrieval.assert_ready()
    identity.assert_tool_calling()
    out.mkdir(parents=True, exist_ok=True)
    queries_path = out / "queries.json"
    queries_path.write_text(json.dumps(list(rows), ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = {
        "harness": harness,
        "harness_mask": dict(harness_mask),
        "served_model": identity.to_dict(),
        "retrieval": retrieval.to_dict(),
        "queries_path": str(queries_path),
        "out": str(out),
        "max_turns": int(max_turns),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "pool_meta": dict(pool_meta),
        "rank": 0,
    }
    if extra:
        cfg.update(dict(extra))
    cfg_path = out / "WORKER_CONFIG.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    env = subprocess_env_for_mask(harness_mask, harness=harness)
    proc = subprocess.run(
        [sys.executable, "-m", "trim.eval.harness1_api_worker", "--config", str(cfg_path)],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        done = out / "DONE.json"
        detail = done.read_text(encoding="utf-8") if done.is_file() else f"exit={proc.returncode}"
        raise RuntimeError(
            "upstream_api eval worker failed; official path does not fall back to "
            f"legacy_local. {detail}"
        )
    traces: list[dict[str, Any]] = []
    pq = out / "PER_QUERY.jsonl"
    if pq.is_file():
        for line in pq.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces.append(json.loads(line))
    summary_path = out / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return summary, traces
