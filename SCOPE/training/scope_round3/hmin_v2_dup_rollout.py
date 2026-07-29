#!/usr/bin/env python3
"""H_min_v2 closed-loop rollout with Dup operation + ActionRealizer (Round 3)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.agent import OpenAIAgentInferenceModel
from harness.harness_config import apply_harness_config, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name, get_llm_settings
from training.chat_decision_driver import ChatDecisionDriver
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.rollout_worker import QueryRecord
from training.opd.vllm_server import start_vllm_server
from training.scope.distillability.metrics import enrich_episode_metrics
from training.scope.dup_operation_runtime import DupOperationRuntime, DupOperationRuntimeConfig
from training.scope.dup_telemetry import DupTelemetryAggregator
from training.train_rl import SlidingWindowSearchEnv


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_shard_queries(
    manifest_path: Path, shard: str, n_shards: int = 8
) -> list[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if shard in data.get("shards", {}):
        return [str(x) for x in data["shards"][shard]]
    qids = [str(x) for x in data.get("query_ids", [])]
    idx = int(shard.replace("shard", ""))
    size = len(qids) // n_shards
    start = idx * size
    end = start + size if idx < n_shards - 1 else len(qids)
    return qids[start:end]


def _load_records(manifest_path: Path, shard_qids: list[str]) -> list[QueryRecord]:
    all_records = {
        r.query_id: r
        for r in load_browsecomp_full_queries(split="all", limit=0, download_if_missing=False)
    }
    return [all_records[qid] for qid in shard_qids if qid in all_records]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    p.add_argument("--shard", type=str, default="shard0")
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument(
        "--harness-config",
        default=str(_REPO / "harness/configs/modules_minimal_v2.yaml"),
    )
    p.add_argument("--max-turns", type=int, default=35)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-port", type=int, default=8900)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--query-timeout-s", type=float, default=600.0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--manage-vllm", action="store_true", default=True)
    p.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--bm25-index-path", default=None)
    p.add_argument(
        "--dup-operation",
        action="store_true",
        default=False,
        help="Enable typed KEEP/SKIP operation + ActionRealizer",
    )
    p.add_argument(
        "--collect-states-only",
        action="store_true",
        default=False,
        help="Base rollout without dup operation (for labeling)",
    )
    return p.parse_args()


def _load_dup_runtime(
    model_path: str,
    adapter_path: Path | None,
    device: str,
    *,
    vllm_client=None,
    vllm_model: str | None = None,
) -> DupOperationRuntime | None:
    if vllm_client is not None and vllm_model:
        from training.scope.vllm_operation_scorer import VllmOperationScorer

        return DupOperationRuntime(
            None,
            None,
            vllm_scorer=VllmOperationScorer(vllm_client, vllm_model),
        )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    if adapter_path and adapter_path.exists():
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return DupOperationRuntime(model, tokenizer, device=dev)


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
    for k, v in {
        "V8D_SUBTRACTIVE_CURATION": "0",
        "V8D_IMPORTANCE_TAGGING": "0",
        "V8D_AUTO_POPULATE_FIRST_SEARCH": "0",
        "V8D_EVIDENCE_GRAPH": "0",
        "V8D_SENTENCE_COMPRESS": "0",
        "V8D_CONTENT_DEDUP": "0",
        "V8D_VERIFY_TOOL": "1",
        "V8D_TOKEN_BUDGET_MARKER": "0",
        "V8D_CHUNK_NEIGHBORS": "0",
    }.items():
        os.environ[k] = v

    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)

    shard_qids = _load_shard_queries(Path(args.manifest), args.shard, args.n_shards)
    records = _load_records(Path(args.manifest), shard_qids)

    resolved = {
        "model_path": args.model_path,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "harness_config": args.harness_config,
        "manifest": str(args.manifest),
        "shard": args.shard,
        "query_ids": shard_qids,
        "dup_operation": args.dup_operation,
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    index_path = check_retrieval_backend("bm25", bm25_index_path=args.bm25_index_path)
    runtime = build_rollout_runtime(
        "browsecompplus",
        collection_split="test",
        reranker="none",
        retrieval="bm25",
        bm25_index_path=index_path,
    )

    episodes_path = out_dir / "episodes.jsonl"
    states_path = out_dir / "decision_states.jsonl"
    telemetry_path = out_dir / "dup_admission_events.jsonl"
    done = load_completed_query_ids(episodes_path) if args.resume else set()
    pending = [r for r in records if r.query_id not in done]

    vllm_handle = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
    os.environ["base_url"] = base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = "hmin-v2-rollout"
    get_llm_settings.cache_clear()

    if args.manage_vllm and args.vllm_url is None:
        vllm_handle = start_vllm_server(
            model_path=args.model_path,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=32768,
            served_model_name="hmin-v2-rollout",
            log_path=str(out_dir / "vllm_server.log"),
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )
        os.environ["base_url"] = vllm_handle.base_url
        get_llm_settings.cache_clear()

    client = get_llm_client()
    model_name = get_llm_model_name()
    inference = OpenAIAgentInferenceModel(
        openai_client=client,
        model=model_name,
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        api_style="chat_completions",
    )

    dup_rt: DupOperationRuntime | None = None
    if args.dup_operation and not args.collect_states_only:
        dup_rt = _load_dup_runtime(
            args.model_path,
            args.adapter_path,
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "cuda",
            vllm_client=client,
            vllm_model=model_name,
        )

    global_telemetry = DupTelemetryAggregator()
    sem = asyncio.Semaphore(args.parallel)
    write_lock = asyncio.Lock()

    async def _one(record: QueryRecord) -> None:
        local_dup = dup_rt
        pre_hook = None
        if local_dup is not None:
            pre_hook = local_dup.make_pre_step_hook(query_id=record.query_id)
        try:
            async with sem:
                env = SlidingWindowSearchEnv(
                    toolset=runtime.toolset,
                    search_tool=runtime.search_tool,
                    query_id=record.query_id,
                    query_text=record.query,
                    dataset=runtime.dataset,
                    text_token_counter=runtime.text_token_counter,
                    max_turns=args.max_turns,
                )
                driver = ChatDecisionDriver(env=env, inference=inference, max_turns=args.max_turns)
                result = await asyncio.wait_for(
                    driver.run(pre_step_hook=pre_hook),
                    timeout=float(args.query_timeout_s),
                )
        except asyncio.TimeoutError:
            result = {
                "query_id": record.query_id,
                "error": True,
                "turns": 0,
                "recall": 0.0,
            }

        ep = enrich_episode_metrics("duplicate_evidence", dict(result))
        ep["query_id"] = record.query_id
        if local_dup is not None:
            tel = local_dup.telemetry.summarize()
            ep.update(
                {
                    "duplicate_curate_rate": tel["duplicate_curate_rate"],
                    "false_skip_rate": tel["false_skip_rate"],
                    "dup_telemetry": tel,
                }
            )
            for ev in local_dup.telemetry.events:
                global_telemetry.add(ev)
        ep.pop("turn_records", None)
        states: list[dict[str, Any]] = []
        for tr in result.get("turn_records") or []:
            ds = tr.decision_state.to_dict() if hasattr(tr, "decision_state") else {}
            states.append(
                {
                    "query_id": record.query_id,
                    "turn_id": tr.turn_id,
                    "decision_state": ds,
                    "student_action": tr.student_action.to_dict()
                    if hasattr(tr.student_action, "to_dict")
                    else {},
                }
            )

        async with write_lock:
            with episodes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            with states_path.open("a", encoding="utf-8") as f:
                for st in states:
                    f.write(json.dumps(st, ensure_ascii=False) + "\n")
            if local_dup is not None:
                with telemetry_path.open("a", encoding="utf-8") as f:
                    for ev in local_dup.telemetry.events:
                        f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")

    await asyncio.gather(*[_one(r) for r in pending])

    if vllm_handle is not None:
        vllm_handle.stop()

    summary = {
        "n_queries": len(shard_qids),
        "n_completed": len(pending),
        "shard": args.shard,
        "dup_operation": args.dup_operation,
        "output_dir": str(out_dir),
        "dup_telemetry": global_telemetry.summarize() if dup_rt else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
