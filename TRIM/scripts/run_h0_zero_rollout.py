#!/usr/bin/env python3
"""H0 zero rollout on the in-distribution BC+ pool. Stops when the score is written.

This is the fixed-base measurement before any harness search or TRIM update.
``--component zero`` stays the live mask. Default hardware matches the 32-GPU
training launch: 4 nodes × 8 replicas, one vLLM actor per card.

On each node the command is the same except ``--node-rank``. Node 0 waits for
the other node shards on the shared ``--out`` directory, merges per-query
traces, writes ``ZERO_ROLLOUT.json``, and exits.

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_h0_zero_rollout.py \\
      --nnodes 4 --rollout-replicas 8 --node-rank $NODE_RANK \\
      --model_name /path/to/base-model \\
      --out /shared/h0_zero_bcplus_full

One 32-GPU machine:

    python TRIM/scripts/run_h0_zero_rollout.py \\
      --nnodes 1 --rollout-replicas 32 \\
      --model_name /path/to/base-model \\
      --out /shared/h0_zero_bcplus_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.eval_parallel import shard_rows_round_robin
from trim.eval.harness1_metrics import summarize_quality_and_timing
from trim.rsi.canonical import sha256_json

DEFAULT_NNODES = 4
DEFAULT_REPLICAS_PER_NODE = 8
DEFAULT_BENCHMARK = "bcplus_full"
DEFAULT_COMPONENT = "zero"


def world_size(nnodes: int, replicas_per_node: int) -> int:
    return int(nnodes) * int(replicas_per_node)


def node_rows(rows: Sequence[dict[str, Any]], *, nnodes: int, replicas_per_node: int, node_rank: int) -> list[dict[str, Any]]:
    shards = shard_rows_round_robin(list(rows), world_size(nnodes, replicas_per_node))
    start = int(node_rank) * int(replicas_per_node)
    owned: list[dict[str, Any]] = []
    for shard in shards[start : start + int(replicas_per_node)]:
        owned.extend(shard)
    return owned


def build_eval_argv(
    *,
    model_name: str,
    out: Path,
    query_ids: Path,
    replicas_per_node: int,
    tensor_parallel_size: int,
    extra: Sequence[str],
) -> list[str]:
    argv = [
        "--harness",
        "Harness-1",
        "--benchmark",
        DEFAULT_BENCHMARK,
        "--component",
        DEFAULT_COMPONENT,
        "--evaluation-path",
        "legacy_local",
        "--model_name",
        model_name,
        "--eval-query-ids",
        str(query_ids),
        "--tp",
        str(int(replicas_per_node)),
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
        "--out",
        str(out),
    ]
    argv.extend(extra)
    return argv


def _parse(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="H0 zero rollout on bcplus_full. Eval only.")
    parser.add_argument("--nnodes", type=int, default=DEFAULT_NNODES)
    parser.add_argument("--rollout-replicas", type=int, default=DEFAULT_REPLICAS_PER_NODE)
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge-timeout-s", type=float, default=48 * 3600)
    parser.add_argument("--plan-only", action="store_true")
    known, extra = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    blocked = {"--component", "--benchmark", "--train_method", "--train-method", "--harness-bank", "--evaluation-path"}
    for token in extra:
        if token in blocked or token.startswith("--component=") or token.startswith("--benchmark="):
            raise SystemExit(f"{token} is fixed by the H0 zero rollout entry")
    if int(known.nnodes) < 1 or int(known.rollout_replicas) < 1 or int(known.tensor_parallel_size) < 1:
        raise SystemExit("nnodes, rollout-replicas, and tensor-parallel-size must be positive")
    if known.node_rank is not None and not 0 <= int(known.node_rank) < int(known.nnodes):
        raise SystemExit(f"--node-rank {known.node_rank} is outside 0..{int(known.nnodes) - 1}")
    return known, extra


def _write_ids(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['query_id']}\n" for row in rows), encoding="utf-8")


def _load_traces(node_dir: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    paths = sorted(node_dir.glob("*/PER_QUERY.jsonl"))
    if not paths and (node_dir / "PER_QUERY.jsonl").is_file():
        paths = [node_dir / "PER_QUERY.jsonl"]
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces.append(json.loads(line))
    return traces


def merge_zero_rollout(node_dirs: Sequence[Path], *, expected_ids: Sequence[str]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for node_dir in node_dirs:
        for trace in _load_traces(node_dir):
            qid = str(trace.get("query_id") or "")
            if not qid:
                continue
            if qid in seen:
                duplicates.append(qid)
                continue
            seen.add(qid)
            traces.append(trace)
    if duplicates:
        raise SystemExit(f"duplicate zero-rollout query ids: {sorted(set(duplicates))[:8]}")
    missing = [qid for qid in expected_ids if qid not in seen]
    summary = summarize_quality_and_timing(traces)
    summary.update(
        {
            "role": "h0_zero_rollout",
            "harness": "Harness-1",
            "component": DEFAULT_COMPONENT,
            "benchmark": DEFAULT_BENCHMARK,
            "n_expected": len(expected_ids),
            "n_queries": len(traces),
            "missing_query_ids": missing[:16],
            "execution_complete": not missing and len(traces) == len(expected_ids),
            "stop_after": "eval",
            "next_stage": None,
        }
    )
    return summary


def _print_plan(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    print(
        json.dumps(
            {
                "role": "h0_zero_rollout",
                "benchmark": DEFAULT_BENCHMARK,
                "component": DEFAULT_COMPONENT,
                "nnodes": int(args.nnodes),
                "rollout_replicas_per_node": int(args.rollout_replicas),
                "tensor_parallel_size": int(args.tensor_parallel_size),
                "world_size": world_size(args.nnodes, args.rollout_replicas),
                "stop_after": "eval",
                "out": str(args.out),
            },
            indent=2,
        ),
        flush=True,
    )
    if int(args.nnodes) > 1 and args.node_rank is None:
        print("Pass --node-rank on each machine. Shared --out is required.", flush=True)
        for rank in range(int(args.nnodes)):
            cmd = [
                sys.executable,
                str(script),
                "--nnodes",
                str(int(args.nnodes)),
                "--rollout-replicas",
                str(int(args.rollout_replicas)),
                "--tensor-parallel-size",
                str(int(args.tensor_parallel_size)),
                "--node-rank",
                str(rank),
                "--model_name",
                args.model_name,
                "--out",
                str(args.out),
            ]
            print(" ".join(cmd), flush=True)


def _wait_for_nodes(out: Path, nnodes: int, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    pending = list(range(nnodes))
    while pending:
        pending = [rank for rank in pending if not (out / "nodes" / f"node{rank}" / "NODE_DONE.json").is_file()]
        if not pending:
            return
        if time.time() > deadline:
            raise SystemExit(f"timed out waiting for zero-rollout nodes {pending}")
        print(f"[h0-zero] waiting for nodes {pending}", flush=True)
        time.sleep(15)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = _parse(argv)
    _print_plan(args)
    if args.plan_only or (int(args.nnodes) > 1 and args.node_rank is None):
        return 0

    from trim.eval.transfer_benchmarks import load_eval_benchmark

    rows, _pool = load_eval_benchmark(DEFAULT_BENCHMARK)
    rank = 0 if args.node_rank is None else int(args.node_rank)
    owned = node_rows(rows, nnodes=int(args.nnodes), replicas_per_node=int(args.rollout_replicas), node_rank=rank)
    if not owned:
        raise SystemExit(f"node {rank} received no bcplus_full queries")
    node_dir = args.out / "nodes" / f"node{rank}"
    id_path = node_dir / "query_ids.txt"
    _write_ids(id_path, owned)
    plan = {
        "benchmark": DEFAULT_BENCHMARK,
        "component": DEFAULT_COMPONENT,
        "n_queries": len(rows),
        "ids_sha256": sha256_json([str(row["query_id"]) for row in rows]),
        "nnodes": int(args.nnodes),
        "rollout_replicas_per_node": int(args.rollout_replicas),
        "world_size": world_size(args.nnodes, args.rollout_replicas),
    }
    (args.out / "PLAN.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    _write_ids(args.out / "ALL_QUERY_IDS.txt", rows)

    eval_main = _eval_main()
    code = int(
        eval_main(
            build_eval_argv(
                model_name=str(args.model_name),
                out=node_dir,
                query_ids=id_path,
                replicas_per_node=int(args.rollout_replicas),
                tensor_parallel_size=int(args.tensor_parallel_size),
                extra=extra,
            )
        )
        or 0
    )
    if code != 0:
        return code
    (node_dir / "NODE_DONE.json").write_text(
        json.dumps({"ok": True, "node_rank": rank, "n_queries": len(owned)}) + "\n",
        encoding="utf-8",
    )
    if rank != 0:
        print(json.dumps({"role": "h0_zero_rollout", "node_rank": rank, "stop_after": "eval", "merged": False}), flush=True)
        return 0
    _wait_for_nodes(args.out, int(args.nnodes), float(args.merge_timeout_s))
    expected = [line.strip() for line in (args.out / "ALL_QUERY_IDS.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    node_dirs = [args.out / "nodes" / f"node{i}" for i in range(int(args.nnodes))]
    summary = merge_zero_rollout(node_dirs, expected_ids=expected)
    (args.out / "ZERO_ROLLOUT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["execution_complete"] else 2


def _eval_main():
    import importlib.util

    path = _TRIM / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("trim_run_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    raise SystemExit(main())
