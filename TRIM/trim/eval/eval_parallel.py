"""Data-parallel eval: shard queries across replica vLLM/HF servers, then merge."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from trim.eval.sr_opd_four_cell_eval import pack_closed_loop_summary, split_summaries


def parse_gpu_ids(explicit: str | None = None) -> list[int]:
    text = str(explicit or "").strip()
    if text:
        return [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    env = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if env and env not in {"-1", "none", "None"}:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    try:
        import torch

        n = int(torch.cuda.device_count() or 0)
        return list(range(n)) if n else [0]
    except Exception:
        return [0]


def replica_tp_size(*, eval_replicas: int, tensor_parallel_size: int | None) -> int:
    if tensor_parallel_size is not None and int(tensor_parallel_size) > 0:
        return int(tensor_parallel_size)
    return 1 if int(eval_replicas) > 1 else 0


def assign_replica_gpus(
    gpu_ids: Sequence[int],
    *,
    n_replicas: int,
    tp_size: int,
) -> list[list[int]]:
    n_replicas = int(n_replicas)
    tp_size = int(tp_size)
    if n_replicas < 1:
        raise ValueError("--tp must be >= 1")
    if tp_size < 1:
        raise ValueError("tensor parallel size must be >= 1")
    need = n_replicas * tp_size
    if len(gpu_ids) < need:
        raise ValueError(
            f"--tp {n_replicas} x tensor-parallel-size {tp_size} needs {need} GPUs, "
            f"got {list(gpu_ids)}"
        )
    assigned: list[list[int]] = []
    for rank in range(n_replicas):
        start = rank * tp_size
        assigned.append([int(x) for x in gpu_ids[start : start + tp_size]])
    return assigned


def shard_rows_round_robin(rows: Sequence[dict[str, Any]], n_shards: int) -> list[list[dict[str, Any]]]:
    n_shards = int(n_shards)
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
    for i, row in enumerate(rows):
        shards[i % n_shards].append(row)
    return shards


def effective_replica_count(n_rows: int, requested: int) -> int:
    return max(1, min(int(requested), max(1, int(n_rows))))


def merge_traces(
    shard_traces: Sequence[Sequence[dict[str, Any]]],
    original_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for traces in shard_traces:
        for tr in traces:
            qid = str(tr.get("query_id") or "")
            if qid:
                by_id[qid] = tr
    missing = [str(r["query_id"]) for r in original_rows if str(r["query_id"]) not in by_id]
    if missing:
        raise RuntimeError(f"merged eval is missing {len(missing)} queries, e.g. {missing[:8]}")
    extra = set(by_id) - {str(r["query_id"]) for r in original_rows}
    if extra:
        raise RuntimeError(f"merged eval has {len(extra)} unexpected queries, e.g. {sorted(extra)[:8]}")
    return [by_id[str(r["query_id"])] for r in original_rows]


def summarize_merged_traces(
    traces: list[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    leak_count: int,
    primary_split: str,
    extra: dict[str, Any] | None = None,
    retrieval_name: str | None = None,
) -> dict[str, Any]:
    name = retrieval_name
    if not name:
        name = str((extra or {}).get("retrieval") or "")
    if not name:
        has_retrieval = any(t.get("evidence_recall_at_5") is not None for t in traces)
        name = "pyserini_lucene" if has_retrieval else "none"
    split = split_summaries(
        traces,
        setting="closed_loop",
        retrieval_name=name,
        eval_rows=list(rows),
    )
    payload = pack_closed_loop_summary(
        split,
        leak=int(leak_count),
        n_rows=len(rows),
        primary_split=primary_split,
        extra=extra,
    )
    payload["n_queries"] = len(traces)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_replicated_eval(
    *,
    rows: list[dict[str, Any]],
    out: Path,
    cell: str,
    adapter_path: str | None,
    spec_out_env: dict[str, Any],
    eval_replicas: int,
    gpu_ids: Sequence[int],
    tensor_parallel_size: int,
    stagger_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_replicas = effective_replica_count(len(rows), eval_replicas)
    gpu_groups = assign_replica_gpus(gpu_ids, n_replicas=n_replicas, tp_size=tensor_parallel_size)
    shards = shard_rows_round_robin(rows, n_replicas)
    plan = []
    procs: list[subprocess.Popen] = []
    logs: list[Any] = []
    pythonpath = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    if pythonpath not in existing.split(os.pathsep):
        existing = pythonpath + (os.pathsep + existing if existing else "")
    try:
        for rank, (shard, gpus) in enumerate(zip(shards, gpu_groups)):
            if not shard:
                continue
            shard_dir = out / "shards" / f"{cell}_rank{rank}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            config = dict(spec_out_env)
            config.update(
                {
                    "rank": rank,
                    "n_replicas": n_replicas,
                    "gpu_ids": list(gpus),
                    "cell": cell,
                    "adapter_path": adapter_path,
                    "out": str(shard_dir),
                }
            )
            queries_path = shard_dir / "queries.json"
            config_path = shard_dir / "SHARD_CONFIG.json"
            write_json(queries_path, shard)
            config["queries_path"] = str(queries_path)
            write_json(config_path, config)
            plan.append(
                {
                    "rank": rank,
                    "n_queries": len(shard),
                    "gpu_ids": list(gpus),
                    "query_ids": [r["query_id"] for r in shard],
                    "out": str(shard_dir),
                }
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = existing
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            log_path = shard_dir / "shard_worker.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            logs.append(log_fh)
            proc = subprocess.Popen(
                [sys.executable, "-m", "trim.eval.eval_shard_worker", "--config", str(config_path)],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            procs.append(proc)
            if stagger_s > 0 and rank + 1 < n_replicas:
                time.sleep(float(stagger_s))
        write_json(out / f"SHARD_PLAN_{cell}.json", {"cell": cell, "shards": plan})
        failures = []
        for proc, shard_meta in zip(procs, plan):
            rc = proc.wait()
            if rc != 0:
                failures.append({**shard_meta, "returncode": rc})
        if failures:
            raise RuntimeError(f"eval shards failed: {failures}")
        shard_traces: list[list[dict[str, Any]]] = []
        leak_count = 0
        for shard_meta in plan:
            shard_dir = Path(shard_meta["out"])
            done = load_json(shard_dir / "DONE.json")
            if not done.get("ok"):
                raise RuntimeError(f"shard {shard_meta['rank']} not ok: {done}")
            shard_traces.append(load_jsonl(shard_dir / "PER_QUERY.jsonl"))
            leak_count += int(done.get("leak_count") or 0)
        traces = merge_traces(shard_traces, rows)
        extra = {
            "eval_replicas": n_replicas,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_ids": [g for group in gpu_groups for g in group],
            "teacher_leak_count": leak_count,
            **{k: spec_out_env[k] for k in ("max_turns", "max_new_tokens", "temperature", "search_k") if k in spec_out_env},
        }
        summary = summarize_merged_traces(
            traces,
            rows,
            leak_count=leak_count,
            primary_split=str(spec_out_env.get("primary_split") or "official_test"),
            extra=extra,
        )
        return summary, traces
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for handle in logs:
            try:
                handle.close()
            except Exception:
                pass
