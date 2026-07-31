#!/usr/bin/env python3
"""BrowseComp bare rollout: tau_i ~ pi_theta(x_i) via vLLM, no Harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.opd.bare_rollout import run_bare_rollout, save_bare_trajectories
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.llm_factory import build_vllm_rollout_backend_from_env, llm_api_configured, llm_manifest_fields
from training.opd.rollout_worker import QueryRecord, load_query_records_from_json
from training.opd.vllm_rollout_backend import VLLMRolloutBackend
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bare BrowseComp rollout (no Harness)")
    parser.add_argument(
        "--model-path",
        default="/data/ppnm/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--queries-json",
        default=None,
        help="Optional debug fixture; default uses full BrowseComp+ dataset",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "test", "rl", "sft"],
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = full split")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Per-query generation budget (Harness-1 BrowseComp uses 2048/turn)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="On-policy sampling temperature (Harness-1 eval uses 1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM context window (prompt + completion)",
    )
    parser.add_argument("--output-dir", default="outputs/bare_rollout_browsecomp_full")
    parser.add_argument("--vllm-port", type=int, default=8770)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--manage-vllm", action="store_true", default=True)
    parser.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Concurrent in-flight bare generation requests to vLLM (default: 8)",
    )
    parser.add_argument(
        "--use-llm-api",
        action="store_true",
        help="Force OpenAI-compatible API from BiSHOP/.env (base_url, api_key, model_name)",
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "bare_rollouts.jsonl"

    records = _load_records(args)
    if not records:
        raise SystemExit("No BrowseComp queries resolved for bare rollout.")

    print(
        f"[bare] Loaded {len(records)} queries "
        f"(split={args.split}, mode=bare, no Harness, parallel={args.parallel})"
    )

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
    use_api = args.use_llm_api or (llm_api_configured() and args.vllm_url is None)

    try:
        if use_api:
            print("[bare] Using LLM API from BiSHOP/.env (base_url, api_key, model_name)")
            rollout = build_vllm_rollout_backend_from_env(tokenizer_path=args.model_path)
            base_url = rollout.base_url
        elif args.vllm_url is None and args.manage_vllm:
            print(f"[bare] Starting vLLM (TP={args.tensor_parallel_size}) at {base_url} ...")
            vllm_handle = start_vllm_server(
                model_path=args.model_path,
                port=args.vllm_port,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                log_path=str(output_dir / "vllm_server.log"),
            )
            base_url = vllm_handle.base_url
            print(f"[bare] vLLM ready: {base_url}")
            rollout = VLLMRolloutBackend(
                base_url=base_url,
                model_name="qwen",
                tokenizer_path=args.model_path,
            )
        else:
            print(f"[bare] Using vLLM at {base_url}")
            rollout = VLLMRolloutBackend(
                base_url=base_url,
                model_name="qwen",
                tokenizer_path=args.model_path,
            )
        run_bare_rollout(
            rollout,
            records,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_jsonl=jsonl_path,
            resume=args.resume,
            parallel=args.parallel,
        )
        path = save_bare_trajectories(
            [],
            output_dir,
            manifest={
                "model_path": args.model_path,
                "vllm_url": base_url,
                "backend": "api" if use_api else "vllm",
                **(llm_manifest_fields() if use_api else {}),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "max_model_len": args.max_model_len,
                "parallel": args.parallel,
                "split": args.split,
                "queries_source": args.queries_json or "browsecompplus_full",
                "resume": args.resume,
            },
        )
        print(f"[bare] Saved trajectories -> {path}")
    finally:
        if vllm_handle is not None:
            print("[bare] Stopping vLLM ...")
            vllm_handle.stop()


if __name__ == "__main__":
    main()
