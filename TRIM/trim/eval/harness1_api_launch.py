"""Dispatch official Harness-1 eval to isolated API workers (E02)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.harness1_api_eval import assert_fresh_eval_dir, summarize_api_traces, write_run_manifest
from trim.eval.eval_parallel import (
    effective_replica_count,
    load_json,
    load_jsonl,
    merge_traces,
    shard_rows_round_robin,
    write_json,
    write_jsonl,
)
from trim.upstream_harness1.model_serve import ServedModelIdentity
from trim.upstream_harness1.retrieval import RetrievalConfig
from trim.upstream_harness1.v8d_flags import subprocess_env_for_mask

_TRIM_ROOT = Path(__file__).resolve().parents[2]


def _worker_config(
    *,
    harness: str,
    harness_mask: Mapping[str, bool],
    identity: ServedModelIdentity,
    retrieval: RetrievalConfig,
    queries_path: Path,
    out: Path,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    pool_meta: Mapping[str, Any],
    rank: int,
    n_replicas: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
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
        "rank": int(rank),
        "n_replicas": int(n_replicas),
    }
    if extra:
        cfg.update(dict(extra))
    return cfg


def _spawn_api_worker(*, cfg_path: Path, env: Mapping[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "trim.eval.harness1_api_worker", "--config", str(cfg_path)],
        cwd=str(_TRIM_ROOT),
        env=dict(env),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc._trim_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _close_worker_logs(procs: Sequence[subprocess.Popen]) -> None:
    for proc in procs:
        handle = getattr(proc, "_trim_log_fh", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


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
    assert_fresh_eval_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    queries_path = out / "queries.json"
    queries_path.write_text(json.dumps(list(rows), ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = _worker_config(
        harness=harness,
        harness_mask=harness_mask,
        identity=identity,
        retrieval=retrieval,
        queries_path=queries_path,
        out=out,
        max_turns=max_turns,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pool_meta=pool_meta,
        rank=0,
        n_replicas=1,
        extra=extra,
    )
    cfg_path = out / "WORKER_CONFIG.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    env = subprocess_env_for_mask(harness_mask, harness=harness)
    proc = subprocess.run(
        [sys.executable, "-m", "trim.eval.harness1_api_worker", "--config", str(cfg_path)],
        cwd=str(_TRIM_ROOT),
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


def run_replicated_api_eval(
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
    eval_replicas: int,
    stagger_s: float = 0.0,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Shard queries across parallel API workers sharing the same actor endpoint."""
    n_replicas = effective_replica_count(len(rows), eval_replicas)
    if n_replicas <= 1:
        return run_isolated_api_eval(
            rows=rows,
            out=out,
            harness=harness,
            harness_mask=harness_mask,
            identity=identity,
            retrieval=retrieval,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pool_meta=pool_meta,
            extra=extra,
        )

    retrieval.assert_ready()
    identity.assert_tool_calling()
    assert_fresh_eval_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    shards = shard_rows_round_robin(list(rows), n_replicas)
    env = subprocess_env_for_mask(harness_mask, harness=harness)
    plan: list[dict[str, Any]] = []
    procs: list[subprocess.Popen] = []
    try:
        for rank, shard in enumerate(shards):
            if not shard:
                continue
            shard_dir = out / "shards" / f"rank{rank}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            queries_path = shard_dir / "queries.json"
            queries_path.write_text(json.dumps(shard, ensure_ascii=False) + "\n", encoding="utf-8")
            cfg = _worker_config(
                harness=harness,
                harness_mask=harness_mask,
                identity=identity,
                retrieval=retrieval,
                queries_path=queries_path,
                out=shard_dir,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pool_meta=pool_meta,
                rank=rank,
                n_replicas=n_replicas,
                extra=extra,
            )
            cfg_path = shard_dir / "WORKER_CONFIG.json"
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            plan.append(
                {
                    "rank": rank,
                    "n_queries": len(shard),
                    "query_ids": [str(r["query_id"]) for r in shard],
                    "out": str(shard_dir),
                }
            )
            procs.append(_spawn_api_worker(cfg_path=cfg_path, env=env, log_path=shard_dir / "worker.log"))
            if stagger_s > 0 and rank + 1 < len([s for s in shards if s]):
                time.sleep(float(stagger_s))

        write_json(
            out / "SHARD_PLAN.json",
            {"eval_replicas": n_replicas, "stagger_s": float(stagger_s), "shards": plan},
        )

        failures: list[dict[str, Any]] = []
        for proc, shard_meta in zip(procs, plan):
            rc = proc.wait()
            if rc != 0:
                failures.append({**shard_meta, "returncode": rc})
        if failures:
            raise RuntimeError(f"upstream_api eval shards failed: {failures}")

        shard_traces: list[list[dict[str, Any]]] = []
        merged_turns: list[dict[str, Any]] = []
        for shard_meta in plan:
            shard_dir = Path(shard_meta["out"])
            done = load_json(shard_dir / "DONE.json")
            if not done.get("ok"):
                raise RuntimeError(f"upstream_api shard {shard_meta['rank']} not ok: {done}")
            shard_traces.append(load_jsonl(shard_dir / "PER_QUERY.jsonl"))
            turns_path = shard_dir / "TURNS.jsonl"
            if turns_path.is_file():
                merged_turns.extend(load_jsonl(turns_path))

        traces = merge_traces(shard_traces, rows)
        write_jsonl(out / "PER_QUERY.jsonl", traces)
        if merged_turns:
            write_jsonl(out / "TURNS.jsonl", merged_turns)

        summary = summarize_api_traces(traces)
        summary["eval_profile"] = retrieval.eval_profile()
        summary["eval_replicas"] = n_replicas
        (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "DONE.json").write_text(
            json.dumps({"ok": True, "n_queries": len(traces), "eval_replicas": n_replicas}) + "\n",
            encoding="utf-8",
        )
        write_run_manifest(
            out,
            mask=harness_mask,
            identity=identity,
            retrieval=retrieval,
            pool_meta=pool_meta,
            extra={
                "eval_replicas": n_replicas,
                "shard_plan": str(out / "SHARD_PLAN.json"),
                **(dict(extra) if extra else {}),
            },
        )
        return summary, list(traces)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        _close_worker_logs(procs)
