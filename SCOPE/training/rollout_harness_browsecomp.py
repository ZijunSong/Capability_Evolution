#!/usr/bin/env python3
"""BrowseComp Harness rollout: multi-turn search agent + Harness modules via vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.harness_config import apply_harness_config, config_path, load_harness_config
from inference.evaluate_harness1_vllm import (
    VllmTokenCompleter,
    eval_single_query,
    summarize_results,
)
from inference.evaluate_harness_api import eval_single_query as eval_single_query_api
from training.opd.llm_factory import llm_manifest_fields, resolve_policy_backend
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.env_factory import build_rollout_runtime, build_smoke_bm25_rollout_runtime
from training.opd.harness_rollout import (
    check_retrieval_backend,
    load_completed_query_ids,
    save_harness_manifest,
)
from training.opd.rollout_worker import QueryRecord, load_query_records_from_json
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server
from training.train_rl import MAX_TURNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BrowseComp+ Harness rollout (multi-turn search + modules)"
    )
    parser.add_argument(
        "--model-path",
        default="/data/ppnm/models/Qwen2.5-7B-Instruct",
        help="HF model served by vLLM (Harmony-compatible checkpoint recommended)",
    )
    parser.add_argument(
        "--harness-config",
        default=str(config_path("modules_full.yaml")),
        help="Harness module YAML (default: full operating point)",
    )
    parser.add_argument("--queries-json", default=None)
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "test", "rl", "sft"],
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = full split")
    parser.add_argument(
        "--collection-split",
        default="test",
        choices=["test", "train", "rl"],
        help="Corpus split (Chroma only; ignored for BM25)",
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--output-dir", default="outputs/harness_rollout_browsecomp_full")
    parser.add_argument("--vllm-port", type=int, default=8771)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--vllm-model-name", default="harness-policy")
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--manage-vllm", action="store_true", default=True)
    parser.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--retrieval",
        default="bm25",
        choices=["bm25", "chroma"],
        help="Retrieval backend (default: local BrowseComp+ BM25 index)",
    )
    parser.add_argument(
        "--bm25-index-path",
        default=None,
        help="Path to Lucene BM25 index (default: external/BrowseComp-Plus/indexes/bm25)",
    )
    parser.add_argument(
        "--reranker",
        default="none",
        choices=["baseten", "vllm", "none"],
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--smoke-retrieval",
        action="store_true",
        help="Use in-memory BM25 corpus (no Java/index/API keys)",
    )
    parser.add_argument(
        "--tools-only",
        action="store_true",
        help="Build retrieval toolstack and exit (no vLLM rollout)",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--policy",
        default="auto",
        choices=["auto", "api", "vllm"],
        help="Policy backend: auto prefers .env LLM API when configured",
    )
    return parser.parse_args()


def _load_records(args: argparse.Namespace) -> list[QueryRecord]:
    if args.queries_json:
        records = load_query_records_from_json(args.queries_json)
        if args.limit > 0:
            return records[: args.limit]
        return records
    return load_browsecomp_full_queries(
        split=args.split,
        limit=args.limit,
        download_if_missing=not args.no_download,
    )


async def _run_rollout(
    *,
    records: list[QueryRecord],
    policy_backend: str,
    policy,
    runtime,
    args: argparse.Namespace,
    output_jsonl: Path,
) -> None:
    done_ids = load_completed_query_ids(output_jsonl) if args.resume else set()
    pending = [r for r in records if r.query_id not in done_ids]
    total = len(records)
    print(
        f"[harness] Pending {len(pending)}/{total} episodes "
        f"(resume={args.resume}, parallel={args.parallel})",
        flush=True,
    )

    sem = asyncio.Semaphore(args.parallel)
    write_lock = asyncio.Lock()
    completed = total - len(pending)

    async def _one(record: QueryRecord) -> None:
        nonlocal completed
        async with sem:
            if policy_backend == "api":
                result = await eval_single_query_api(
                    record.query_id,
                    runtime.dataset,
                    runtime.toolset,
                    runtime.search_tool,
                    runtime.text_token_counter,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_trajectory_length=args.max_turns,
                )
            else:
                result = await eval_single_query(
                    record.query_id,
                    runtime.dataset,
                    runtime.toolset,
                    runtime.search_tool,
                    runtime.text_token_counter,
                    policy,
                    args.max_turns,
                )
            result["query"] = record.query
            result["mode"] = "harness"
            result["retrieval"] = args.retrieval
            async with write_lock:
                completed += 1
                with output_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                if args.log_every > 0 and completed % args.log_every == 0:
                    print(
                        f"[harness] progress {completed}/{total} "
                        f"last_qid={record.query_id} "
                        f"recall={result.get('recall', 0):.3f} "
                        f"turns={result.get('turns', 0)} "
                        f"error={result.get('error', False)}",
                        flush=True,
                    )

    await asyncio.gather(*[_one(record) for record in pending])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "harness_rollouts.jsonl"

    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)
    harness_cfg.save_resolved(output_dir / "harness_resolved_config.yaml")

    index_path = args.bm25_index_path
    index_path = args.bm25_index_path
    if not args.skip_preflight:
        print(
            f"[harness] Preflight: checking {args.retrieval} retrieval backend ...",
            flush=True,
        )
        index_path = check_retrieval_backend(
            args.retrieval,
            bm25_index_path=args.bm25_index_path,
            smoke=args.smoke_retrieval,
        )
        print(f"[harness] Retrieval OK ({args.retrieval})", flush=True)
    elif args.smoke_retrieval:
        index_path = "memory://smoke"

    records = _load_records(args)
    if not records:
        raise SystemExit("No BrowseComp queries resolved for harness rollout.")

    print(
        f"[harness] Loaded {len(records)} queries "
        f"(split={args.split}, retrieval={args.retrieval}, "
        f"config={Path(args.harness_config).name})",
        flush=True,
    )

    policy_backend = resolve_policy_backend(
        policy=args.policy,
        manage_vllm=args.manage_vllm,
        vllm_url=args.vllm_url,
    )
    if policy_backend == "vllm":
        print(
            "[harness] NOTE: vLLM policy uses Harmony token rendering. "
            "For external chat APIs (e.g. Kimi), use --policy api or configure "
            "base_url/api_key/model_name in .env with --no-manage-vllm.",
            flush=True,
        )
    else:
        print(
            "[harness] Using OpenAI-compatible chat API from BiSHOP/.env",
            flush=True,
        )

    if args.tools_only:
        if args.smoke_retrieval:
            runtime = build_smoke_bm25_rollout_runtime("browsecompplus")
        else:
            runtime = build_rollout_runtime(
                "browsecompplus",
                collection_split=args.collection_split,
                reranker=args.reranker,
                retrieval=args.retrieval,
                bm25_index_path=index_path,
            )
        text, meta = runtime.search_tool({"query": records[0].query})
        print(
            f"[harness] tools-only OK: retrieval="
            f"{'smoke' if args.smoke_retrieval else args.retrieval}, "
            f"hits={len(meta.returned_chunk_ids) if meta else 0}",
            flush=True,
        )
        return

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
    policy = None

    try:
        if policy_backend == "vllm":
            if args.vllm_url is None and args.manage_vllm:
                print(
                    f"[harness] Starting vLLM (TP={args.tensor_parallel_size}) at {base_url} ...",
                    flush=True,
                )
                vllm_handle = start_vllm_server(
                    model_path=args.model_path,
                    port=args.vllm_port,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    served_model_name=args.vllm_model_name,
                    log_path=str(output_dir / "vllm_server.log"),
                )
                base_url = vllm_handle.base_url
                print(f"[harness] vLLM ready: {base_url}", flush=True)
            else:
                print(f"[harness] Using vLLM at {base_url}", flush=True)

        if args.smoke_retrieval:
            runtime = build_smoke_bm25_rollout_runtime("browsecompplus")
        else:
            runtime = build_rollout_runtime(
                "browsecompplus",
                collection_split=args.collection_split,
                reranker=args.reranker,
                retrieval=args.retrieval,
                bm25_index_path=index_path,
            )
        if policy_backend == "vllm":
            policy = VllmTokenCompleter(
                base_url=base_url,
                model=args.vllm_model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=900,
            )

        asyncio.run(
            _run_rollout(
                records=records,
                policy_backend=policy_backend,
                policy=policy,
                runtime=runtime,
                args=args,
                output_jsonl=jsonl_path,
            )
        )

        results = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = summarize_results(results)
        print(f"[harness] Summary: {json.dumps(summary, indent=2)}", flush=True)

        save_harness_manifest(
            output_dir,
            manifest={
                "model_path": args.model_path,
                "policy_backend": policy_backend,
                "vllm_url": base_url if policy_backend == "vllm" else None,
                **(llm_manifest_fields() if policy_backend == "api" else {}),
                "harness_config": args.harness_config,
                "retrieval": "smoke" if args.smoke_retrieval else args.retrieval,
                "bm25_index_path": index_path if args.retrieval == "bm25" else None,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "max_model_len": args.max_model_len,
                "split": args.split,
                "collection_split": args.collection_split,
                "parallel": args.parallel,
                "reranker": args.reranker,
                "resume": args.resume,
                "summary": summary,
            },
        )
        print(f"[harness] Done -> {jsonl_path}", flush=True)
    finally:
        if vllm_handle is not None:
            print("[harness] Stopping vLLM ...", flush=True)
            vllm_handle.stop()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
