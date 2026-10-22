#!/usr/bin/env python3
"""Self-evolve a harness bank on frozen H0 weights. Stop before RL/OPD.

The completed zero rollout is the baseline. Each generation asks the same
model for one component edit, then scores H0+bank against that baseline on
bcplus_full with the 32-GPU layout. The process exits when the search policy
says to train, or when it stops with no trainable teacher. It does not launch
TRIM.

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_harness_evolve.py \\
      --nnodes 4 --rollout-replicas 8 --node-rank $NODE_RANK \\
      --model_name /path/to/base-model \\
      --zero-rollout-dir /shared/h0_zero_bcplus_full \\
      --out /shared/h0_harness_search
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.evolution.campaign import search_until_trim
from trim.evolution.diagnosis import cluster_failures
from trim.evolution.self_proposer import SelfHarnessProposer
from trim.rsi.canonical import sha256_json

DEFAULT_NNODES = 4
DEFAULT_REPLICAS_PER_NODE = 8


def _zero_module():
    path = _TRIM / "scripts" / "run_h0_zero_rollout.py"
    spec = importlib.util.spec_from_file_location("run_h0_zero_rollout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval_main():
    path = _TRIM / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("trim_run_eval_for_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _parse(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evolve a harness on frozen H0 and stop before TRIM.")
    parser.add_argument("--nnodes", type=int, default=DEFAULT_NNODES)
    parser.add_argument("--rollout-replicas", type=int, default=DEFAULT_REPLICAS_PER_NODE)
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--evaluation-path",
        choices=("legacy_local", "upstream_api"),
        default="legacy_local",
        help=(
            "Candidate evaluation runtime. Qwen3.5 requires upstream_api; "
            "legacy_local uses the in-process LLM API and cannot parse Qwen3.5 tool calls."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI-compatible actor URL(s) for upstream_api candidate evaluation.",
    )
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--verify-base-url",
        default=None,
        help="OpenAI-compatible verifier URL for component=all (verify_tool). "
        "run_harness_evolve scores with --component all, whose mask sets verify_tool=True; "
        "env_bridge raises if verify_tool is on but no verifier is provided.",
    )
    parser.add_argument(
        "--verify-model",
        default=None,
        help="Verifier served-model name matching --verify-base-url (e.g. harness-1-verifier).",
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--zero-rollout-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-generations", type=int, default=3)
    parser.add_argument("--max-generations", type=int, default=8)
    parser.add_argument("--max-candidate-evaluations", type=int, default=16)
    parser.add_argument("--merge-timeout-s", type=float, default=48 * 3600)
    parser.add_argument("--plan-only", action="store_true")
    known, extra = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    for token in extra:
        if token in {"--train_method", "--train-method", "--component", "--benchmark"}:
            raise SystemExit(f"{token} is not set on the harness-search entry")
    if known.node_rank is not None and not 0 <= int(known.node_rank) < int(known.nnodes):
        raise SystemExit(f"--node-rank {known.node_rank} is outside 0..{int(known.nnodes) - 1}")
    if known.evaluation_path == "upstream_api" and int(known.nnodes) != 1:
        raise SystemExit(
            "--evaluation-path upstream_api currently requires --nnodes 1: "
            "candidate actor servers are local to the single-node eval process"
        )
    return known, extra


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_upstream_baseline(zero_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    api_dir = zero_dir / "upstream_api"
    summary_path = api_dir / "SUMMARY.json"
    done_path = api_dir / "DONE.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; finish the upstream_api H0 rollout first")
    summary = _read_json(summary_path)
    done = _read_json(done_path) if done_path.is_file() else {}
    expected = int(summary.get("n_planned") or done.get("n_planned") or 0)
    completed = int(summary.get("n_queries") or done.get("n_queries") or 0)
    complete = bool(summary.get("execution_complete")) and not bool(summary.get("partial"))
    if done:
        complete = complete and bool(done.get("execution_complete")) and not bool(done.get("partial"))
    if expected and completed != expected:
        complete = False
    if not complete:
        raise SystemExit(
            f"refusing incomplete upstream_api baseline {api_dir}: "
            f"completed={completed} planned={expected} partial={summary.get('partial', done.get('partial'))}"
        )
    traces = _load_jsonl(api_dir / "PER_QUERY.jsonl")
    if expected and len(traces) != expected:
        raise SystemExit(f"upstream_api baseline trace count mismatch: {len(traces)} != {expected}")
    return summary, traces


def load_zero_baseline(zero_dir: Path) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    upstream_summary = zero_dir / "upstream_api" / "SUMMARY.json"
    if upstream_summary.is_file():
        summary, traces = _load_upstream_baseline(zero_dir)
    else:
        summary_path = zero_dir / "ZERO_ROLLOUT.json"
        if not summary_path.is_file():
            raise SystemExit(
                f"missing {summary_path} or {upstream_summary}; finish the H0 zero rollout first"
            )
        summary = _read_json(summary_path)
        if summary.get("component") not in {None, "zero"}:
            raise SystemExit("zero rollout summary is not an H0 zero run")
        if not summary.get("execution_complete", False):
            raise SystemExit("refusing incomplete legacy_local H0 zero rollout")
        zero_mod = _zero_module()
        traces = []
        node_root = zero_dir / "nodes"
        if node_root.is_dir():
            for node_dir in sorted(node_root.glob("node*")):
                traces.extend(zero_mod._load_traces(node_dir))
        expected = int(summary.get("n_expected") or 0)
        if expected and len(traces) != expected:
            raise SystemExit(f"legacy H0 trace count mismatch: {len(traces)} != {expected}")
    by_id: dict[str, float] = {}
    for trace in traces:
        qid = str(trace.get("query_id") or "")
        if not qid:
            continue
        value = trace.get("final_answer_recall")
        if value is None:
            value = trace.get("recall")
        by_id[qid] = float(value or 0.0)
    quality = summary.get("final_answer_recall")
    if quality is None:
        if not by_id:
            raise SystemExit("zero rollout has no final_answer_recall and no per-query traces")
        quality = sum(by_id.values()) / len(by_id)
    return float(quality), by_id, traces


def _print_plan(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    world = int(args.nnodes) * int(args.rollout_replicas)
    print(
        json.dumps(
            {
                "role": "harness_self_evolution",
                "benchmark": "bcplus_full",
                "weights": "frozen",
                "nnodes": int(args.nnodes),
                "rollout_replicas_per_node": int(args.rollout_replicas),
                "world_size": world,
                "stop_after": "harness_search",
                "started_trim": False,
                "zero_rollout_dir": str(args.zero_rollout_dir),
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
                "--node-rank",
                str(rank),
                "--model_name",
                args.model_name,
                "--zero-rollout-dir",
                str(args.zero_rollout_dir),
                "--out",
                str(args.out),
            ]
            print(" ".join(cmd), flush=True)


def _wait_nodes(gen_dir: Path, nnodes: int, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    pending = list(range(nnodes))
    while pending:
        pending = [rank for rank in pending if not (gen_dir / "nodes" / f"node{rank}" / "NODE_DONE.json").is_file()]
        if not pending:
            return
        if time.time() > deadline:
            raise SystemExit(f"timed out waiting for harness-eval nodes {pending}")
        print(f"[harness-search] waiting for nodes {pending}", flush=True)
        time.sleep(15)


def _eval_trace_root(node_dir: Path) -> Path:
    """Return the canonical per-query output root for either eval runtime."""
    upstream = node_dir / "upstream_api"
    return upstream if (upstream / "PER_QUERY.jsonl").is_file() else node_dir


def _load_eval_traces(node_dir: Path) -> list[dict[str, Any]]:
    root = _eval_trace_root(node_dir)
    if (root / "PER_QUERY.jsonl").is_file():
        return _load_jsonl(root / "PER_QUERY.jsonl")
    zero_mod = _zero_module()
    return zero_mod._load_traces(node_dir)


def _eval_shard(args: argparse.Namespace, extra: Sequence[str], *, bank_dir: Path, gen_dir: Path, node_rank: int, rows: list[dict]) -> None:
    zero_mod = _zero_module()
    owned = zero_mod.node_rows(
        rows,
        nnodes=int(args.nnodes),
        replicas_per_node=int(args.rollout_replicas),
        node_rank=node_rank,
    )
    node_dir = gen_dir / "nodes" / f"node{node_rank}"
    id_path = node_dir / "query_ids.txt"
    zero_mod._write_ids(id_path, owned)
    evaluation_path = str(args.evaluation_path)
    if evaluation_path == "upstream_api" and not args.api_base_url:
        raise SystemExit(
            "--evaluation-path upstream_api requires --api-base-url; "
            "run vLLM serve actors separately and pass their /v1 URL(s)"
        )
    # component=all mask sets verify_tool=True; env_bridge raises if verify_tool is on
    # but no verifier is provided. Require --verify-base-url for the all-presets eval.
    if not args.verify_base_url:
        raise SystemExit(
            "--component all sets verify_tool=True; --verify-base-url is required. "
            "Start a verifier serve (harness-1) and pass its /v1 URL."
        )
    argv = [
        "--harness",
        "Harness-1",
        "--benchmark",
        "bcplus_full",
        "--component",
        "all",
        "--evaluation-path",
        evaluation_path,
        "--harness-bank",
        str(bank_dir),
        "--model_name",
        str(args.model_name),
        "--eval-query-ids",
        str(id_path),
        "--tp",
        str(int(args.rollout_replicas)),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
        "--out",
        str(node_dir),
    ]
    if args.api_base_url:
        argv.extend(["--api-base-url", str(args.api_base_url)])
    if args.api_model:
        argv.extend(["--api-model", str(args.api_model)])
    if args.api_key:
        argv.extend(["--api-key", str(args.api_key)])
    if args.verify_base_url:
        argv.extend(["--verify-base-url", str(args.verify_base_url)])
    if args.verify_model:
        argv.extend(["--verify-model", str(args.verify_model)])
    argv.extend(list(extra))
    code = int(_eval_main()(argv) or 0)
    if code != 0:
        raise SystemExit(code)
    trace_root = _eval_trace_root(node_dir)
    summary = _read_json(trace_root / "SUMMARY.json") if (trace_root / "SUMMARY.json").is_file() else {}
    done = _read_json(trace_root / "DONE.json") if (trace_root / "DONE.json").is_file() else {}
    complete = bool(summary.get("execution_complete", done.get("execution_complete", False)))
    n_queries = int(summary.get("n_queries", done.get("n_queries", 0)) or 0)
    if not complete or n_queries != len(owned) or summary.get("partial") or done.get("partial"):
        raise SystemExit(
            f"candidate eval incomplete for node {node_rank}: "
            f"completed={n_queries} expected={len(owned)} partial={summary.get('partial', done.get('partial'))}"
        )
    _write_json(node_dir / "NODE_DONE.json", {"ok": True, "node_rank": node_rank, "n_queries": len(owned)})


def _generate_text(model_name: str, prompt: str, *, session_dir: Path, api_base_url: str | None = None, api_model: str | None = None) -> str:
    """Generate text for the proposer. If api_base_url is provided (upstream_api mode),
    use the already-running serve actors via OpenAI HTTP client instead of starting a
    new VLLMGenerateClient (which would conflict with the serve actors for GPU memory)."""
    if api_base_url:
        import urllib.request
        import urllib.error

        # Try each actor URL in order; skip failed ones (some actors may have timed out
        # during serve startup, especially for large models like 27B where GPU0 often fails).
        urls = [u.strip().rstrip("/") + "/chat/completions" for u in api_base_url.split(",")]
        model = api_model or model_name
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
            "seed": 0,
        }).encode("utf-8")
        last_exc = None
        for url in urls:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return str((body.get("choices") or [{}])[0].get("message", {}).get("content") or "")
            except urllib.error.HTTPError as exc:
                # 400/500 from vLLM (e.g., tool-call-parser expects tools but proposer sends none)
                resp_body = ""
                try:
                    resp_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_exc = RuntimeError(f"HTTP {exc.code} from {url}: {resp_body}")
                continue
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"proposer HTTP call failed on all {len(urls)} actor URLs; last error: {last_exc}") from last_exc

    # legacy_local fallback: start a dedicated VLLMGenerateClient (only when no serve actors)
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.vllm_hybrid import GenerateRequest, VLLMGenerateClient

    enc = load_model_encoding(model_name)
    token_ids = list(enc.encode(prompt))
    client = VLLMGenerateClient(
        model_path=model_name,
        session_dir=session_dir,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    client.start()
    try:
        result = client.generate_batch(
            [
                GenerateRequest(
                    request_id="propose",
                    prompt_token_ids=token_ids,
                    max_new_tokens=512,
                    temperature=0.0,
                    seed=0,
                )
            ]
        )[0]
    finally:
        client.close()
    return str(result.text or "")


def _publish_state(out: Path, payload: dict[str, Any]) -> None:
    payload["state_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "state_sha256"})
    _write_json(out / "SEARCH_STATE.json", payload)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = _parse(argv)
    _print_plan(args)
    if args.plan_only or (int(args.nnodes) > 1 and args.node_rank is None):
        return 0
    baseline, zero_by_id, zero_traces = load_zero_baseline(args.zero_rollout_dir)
    from trim.eval.transfer_benchmarks import load_eval_benchmark

    rows, _pool = load_eval_benchmark("bcplus_full")
    rank = 0 if args.node_rank is None else int(args.node_rank)
    if rank != 0:
        return _worker(args, extra, rows)
    clusters = cluster_failures(zero_traces)
    counts = {name: len(items) for name, items in clusters.items()}
    trajectory_id = "h0-zero"
    for items in clusters.values():
        if items and items[0].get("trajectory_id"):
            trajectory_id = str(items[0]["trajectory_id"])
            break
    proposer = SelfHarnessProposer(
        generate_text=lambda prompt: _generate_text(
            str(args.model_name), prompt, session_dir=args.out / "proposer_vllm",
            api_base_url=str(args.api_base_url) if args.api_base_url else None,
            api_model=str(args.api_model) if args.api_model else None,
        ),
        work_dir=args.out / "candidates",
        failure_counts=counts or {"bad_query": 1},
        trajectory_id=trajectory_id,
    )

    def score(proposal):
        gen_dir = Path(proposal.bank_dir).parent
        _write_json(gen_dir / "READY.json", {"bank_dir": proposal.bank_dir, "bank_sha256": proposal.content_hash})
        _eval_shard(args, extra, bank_dir=Path(proposal.bank_dir), gen_dir=gen_dir, node_rank=0, rows=rows)
        _wait_nodes(gen_dir, int(args.nnodes), float(args.merge_timeout_s))
        traces = []
        for node in range(int(args.nnodes)):
            traces.extend(_load_eval_traces(gen_dir / "nodes" / f"node{node}"))
        if len(traces) != len(rows):
            raise SystemExit(
                f"candidate merge is incomplete: got {len(traces)} traces for {len(rows)} queries"
            )
        from trim.evolution.campaign import score_against_zero

        return score_against_zero(proposal, traces, zero_by_id)

    _publish_state(args.out, {"decision": "EVOLVING", "generation": 0, "started_trim": False})
    result = search_until_trim(
        proposer=proposer,
        score_fn=score,
        baseline_quality=baseline,
        min_generations=int(args.min_generations),
        max_generations=int(args.max_generations),
        max_candidate_evaluations=int(args.max_candidate_evaluations),
    )
    _publish_state(args.out, result)
    name = "READY_FOR_TRIM.json" if result.get("ready_for_trim") else "SEARCH_STOPPED.json"
    _write_json(args.out / name, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def _worker(args: argparse.Namespace, extra: Sequence[str], rows: list[dict]) -> int:
    seen: set[str] = set()
    deadline = time.time() + float(args.merge_timeout_s)
    while time.time() < deadline:
        state_path = args.out / "SEARCH_STATE.json"
        if state_path.is_file():
            state = _read_json(state_path)
            if state.get("decision") not in {None, "EVOLVING"}:
                print(
                    json.dumps(
                        {
                            "role": "harness_worker",
                            "node_rank": args.node_rank,
                            "decision": state.get("decision"),
                            "started_trim": False,
                        }
                    ),
                    flush=True,
                )
                return 0
        for ready in sorted((args.out / "candidates").glob("g*/READY.json")) if (args.out / "candidates").is_dir() else []:
            key = str(ready)
            if key in seen:
                continue
            payload = _read_json(ready)
            gen_dir = ready.parent
            _eval_shard(args, extra, bank_dir=Path(payload["bank_dir"]), gen_dir=gen_dir, node_rank=int(args.node_rank), rows=rows)
            seen.add(key)
        time.sleep(10)
    raise SystemExit("harness-search worker timed out")


if __name__ == "__main__":
    raise SystemExit(main())
