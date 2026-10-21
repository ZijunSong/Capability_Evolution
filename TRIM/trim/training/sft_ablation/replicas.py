"""One env-worker subprocess per rollout replica, each with its own GPUs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from trim.adapters.components import zero_mask
from trim.training.sft_ablation.geometry import planned_visible_ids, replica_gpu_plan
from trim.training.sft_ablation.io_util import atomic_write_json, read_json
from trim.upstream_harness1.v8d_flags import all_enabled_mask


def run_replica_jobs(args: Any, dist: Any, *, task: str, mask_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    replicas = max(1, int(os.environ.get("TRIM_ROLLOUT_REPLICAS") or 1))
    width = int(args.tensor_parallel_size)
    plan = replica_gpu_plan(
        planned_visible_ids(replicas, width),
        replicas=replicas,
        tensor_parallel=width,
    )
    mask = all_enabled_mask("Harness-1") if mask_name == "full" else zero_mask("Harness-1")
    records = list(payload.get("records") or [])
    queries = list(payload.get("queries") or [])
    if records:
        shards = [records[index::len(plan)] for index in range(len(plan))]
    elif not queries and task in {"collect", "rollout", "replay"}:
        return [{"ok": True, "archive": {"trajectories": []}, "points": [], "rows": []}]
    else:
        shards = [queries[index::len(plan)] for index in range(len(plan))] if queries else [[] for _ in plan]
    from trim.training.sft_ablation.env_worker import start_env_worker, stop_env_worker

    results: list[dict[str, Any] | None] = []
    root = Path(args.out) / "replica_jobs" / task
    round_id = payload.get("round_id")
    if round_id is not None:
        root = root / f"round-{int(round_id)}"
    root = root / f"rank{int(dist.rank)}"
    pending: list[tuple[int, Any, Path]] = []
    for index, (gpus, shard) in enumerate(zip(plan, shards)):
        if (records or queries) and not shard:
            results.append({"ok": True, "archive": {"trajectories": []}, "points": [], "rows": []})
            continue
        job_dir = root / f"replica-{index}"
        job_dir.mkdir(parents=True, exist_ok=True)
        result_path = job_dir / "result.json"
        if result_path.exists():
            result_path.unlink()
        job = {
            **payload,
            "task": task,
            "mask": mask,
            "queries": [item.get("query_id") for item in shard] if records else shard,
            "records": shard if records else [],
            "trajectories": shard if records else payload.get("trajectories") or [],
            "result": str(result_path),
            "model_name": args.model_name,
            "max_model_len": int(args.max_model_len),
            "max_new_tokens": int(args.max_new_tokens),
            "group_size": int(args.group_size),
            "max_turns": int(args.max_turns),
            "seed": int(args.seed) + int(dist.rank) * 1000 + index,
            "train_env": args.train_env,
            "tensor_parallel_size": width,
            "session_dir": str(job_dir / "session"),
            "adapter": str(payload.get("adapter") or args.init_adapter or ""),
            "max_anchor_scan_events": int(getattr(args, "max_anchor_scan_events", 8) or 8),
            "round_id": round_id,
        }
        job_path = job_dir / "job.json"
        atomic_write_json(job_path, job)
        proc = start_env_worker(
            job_path,
            mask=mask,
            extra_env={"CUDA_VISIBLE_DEVICES": ",".join(gpus)},
        )
        pending.append((index, proc, result_path))
        results.append(None)
    failed: tuple[int, int] | None = None
    for index, proc, _result_path in pending:
        code = int(proc.wait())
        if code != 0 and failed is None:
            failed = (index, code)
            for other_index, other, _path in pending:
                if other_index != index:
                    stop_env_worker(other)
    if failed is not None:
        index, code = failed
        raise RuntimeError(f"{task} replica {index} exited {code}")
    for index, _proc, result_path in pending:
        result = read_json(result_path)
        if not result.get("ok", False):
            raise RuntimeError(f"{task} replica {index} failed: {result.get('error')}")
        results[index] = result
    return [item for item in results if item is not None]


def dump_job(path: Path, job: dict[str, Any]) -> None:
    path.write_text(json.dumps(job), encoding="utf-8")
