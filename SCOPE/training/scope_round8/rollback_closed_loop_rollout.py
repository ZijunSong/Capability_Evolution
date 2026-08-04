#!/usr/bin/env python3
"""Round 8 Phase 3 closed-loop rollout with rollback hard executor."""

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

import yaml

from harness.agent import OpenAIAgentInferenceModel
from harness.harness_config import apply_harness_config, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name, get_llm_settings
from training.chat_decision_driver import ChatDecisionDriver
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.rollout_worker import QueryRecord
from training.opd.vllm_server import start_vllm_server
from training.scope.distillability.metrics import enrich_episode_metrics
from training.scope.rollback_operation_runtime import (
    RollbackOperationRuntime,
    RollbackOperationRuntimeConfig,
)
from training.scope.vllm_rollback_scorer import VllmRollbackScorer
from training.scope_round8.agent_core_rollout import _load_records
from training.scope_round3.hmin_v2_dup_rollout import _load_shard_queries
from training.train_rl import SlidingWindowSearchEnv

HINT = (
    "Hint: if recent queries repeat or evidence stalls, prefer ROLLBACK_TO "
    "a prior checkpoint instead of continuing the failing branch."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", default="base")
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    p.add_argument("--shard", type=str, default="shard0")
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--merged-path", type=Path, default=None)
    p.add_argument(
        "--harness-config",
        default=str(_REPO / "harness/configs/agent_core_recovery.yaml"),
    )
    p.add_argument("--max-turns", type=int, default=35)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-port", type=int, default=9400)
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--query-timeout-s", type=float, default=600.0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--rollback-operation", action="store_true", default=False)
    p.add_argument("--hint-distill", action="store_true", default=False)
    p.add_argument("--soft-replan-only", action="store_true", default=False)
    p.add_argument("--endorse-only", action="store_true", default=False)
    return p.parse_args()


def _apply_env() -> None:
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


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_env()

    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)

    shard_qids = _load_shard_queries(Path(args.manifest), args.shard, args.n_shards)
    records = _load_records(Path(args.manifest), shard_qids)

    model_for_vllm = str(args.merged_path or args.model_path)
    resolved = {
        "variant": args.variant,
        "model_path": args.model_path,
        "merged_path": str(args.merged_path) if args.merged_path else None,
        "harness_config": args.harness_config,
        "manifest": str(args.manifest),
        "shard": args.shard,
        "rollback_operation": args.rollback_operation,
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    index_path = check_retrieval_backend("bm25", bm25_index_path=None)
    runtime = build_rollout_runtime(
        "browsecompplus",
        collection_split="test",
        reranker="none",
        retrieval="bm25",
        bm25_index_path=index_path,
    )

    episodes_path = out_dir / "episodes.jsonl"
    events_path = out_dir / "rollback_events.jsonl"
    done = load_completed_query_ids(episodes_path) if args.resume else set()
    pending = [r for r in records if r.query_id not in done]

    vllm_handle = start_vllm_server(
        model_path=model_for_vllm,
        port=args.vllm_port,
        tensor_parallel_size=1,
        max_model_len=32768,
        served_model_name="rollback-closed-loop",
        log_path=str(out_dir / "vllm_server.log"),
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    os.environ["base_url"] = vllm_handle.base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = "rollback-closed-loop"
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

    rollback_rt: RollbackOperationRuntime | None = None
    if args.rollback_operation:
        hint = HINT if args.hint_distill else ""
        scorer = VllmRollbackScorer(
            client=client,
            model=model_name,
            model_path=model_for_vllm,
            hint=hint,
        )
        rollback_rt = RollbackOperationRuntime(
            config=RollbackOperationRuntimeConfig(
                enabled=True,
                threshold=0.0,
                soft_replan_only=args.soft_replan_only,
                hint=hint,
                checkpoint_label=args.variant,
            ),
            vllm_scorer=scorer,
        )

    sem = asyncio.Semaphore(args.parallel)
    write_lock = asyncio.Lock()
    recalls: list[float] = []

    async def _one(record: QueryRecord) -> None:
        ctx = None
        pre_hook = None
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
                if rollback_rt is not None:
                    ctx = rollback_rt.fork_for_query(
                        env, query_id=record.query_id, max_turns=args.max_turns
                    )
                    pre_hook = ctx.make_pre_step_hook()
                driver = ChatDecisionDriver(
                    env=env, inference=inference, max_turns=args.max_turns
                )
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
        except Exception as exc:
            result = {
                "query_id": record.query_id,
                "error": True,
                "error_message": str(exc),
                "turns": 0,
                "recall": 0.0,
            }

        ep = enrich_episode_metrics("rollback_decision", dict(result))
        ep["query_id"] = record.query_id
        ep["variant"] = args.variant
        if ctx is not None:
            ep["rollback_telemetry"] = {
                "n_events": len(ctx.events),
                "invalid_checkpoint_predictions": rollback_rt.invalid_checkpoint_predictions,
                "budget_violations": rollback_rt.budget_violations,
            }
        ep.pop("turn_records", None)
        recalls.append(float(ep.get("recall", 0.0)))

        event_rows: list[dict[str, Any]] = []
        if ctx is not None:
            for ev in ctx.events:
                if args.endorse_only and ev.route != "ENDORSE":
                    continue
                event_rows.append(
                    {
                        "query_id": ev.query_id,
                        "turn_id": ev.turn_id,
                        "student_operation": ev.student_operation,
                        "shadow_operation": ev.shadow_operation,
                        "shadow_checkpoint_id": ev.shadow_checkpoint_id,
                        "predicted_checkpoint_id": ev.predicted_checkpoint_id,
                        "route": ev.route,
                        "rollback_success": ev.rollback_success,
                        "state_hash_restore": ev.state_hash_restore,
                    }
                )

        async with write_lock:
            with episodes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            if event_rows:
                with events_path.open("a", encoding="utf-8") as f:
                    for row in event_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*[_one(r) for r in pending])
    vllm_handle.stop()

    summary = {
        "variant": args.variant,
        "shard": args.shard,
        "n_queries": len(shard_qids),
        "n_completed": len(recalls),
        "mean_recall": sum(recalls) / max(1, len(recalls)),
        "output_dir": str(out_dir),
    }
    if rollback_rt is not None:
        summary["rollback_runtime"] = {
            "n_events": len(rollback_rt.events),
            "invalid_checkpoint_predictions": rollback_rt.invalid_checkpoint_predictions,
            "budget_violations": rollback_rt.budget_violations,
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
