#!/usr/bin/env python3
"""Collect natural + injected rollback decision states (Round 8 Phase 1)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.agent import OpenAIAgentInferenceModel
from harness.capability.rollback_operation import RollbackReasonCode
from harness.capability.state import DecisionState
from harness.harness_config import apply_harness_config, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name, get_llm_settings
from harness.recovery.checkpoint_store import CheckpointStore
from harness.recovery.stagnation_detector import FailureEvent, StagnationDetector
from training.chat_decision_driver import ChatDecisionDriver
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend
from training.opd.vllm_server import start_vllm_server
from training.scope.rollback_decision_state import build_rollback_decision_state
from training.scope.rollback_shadow import RollbackBilateralShadow
from training.scope_round8.agent_core_rollout import _load_records
from training.scope_round3.hmin_v2_dup_rollout import _load_shard_queries
from training.train_rl import SlidingWindowSearchEnv

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    p.add_argument("--shard", default="shard0")
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument("--mode", choices=["natural", "injected"], default="natural")
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--harness-config",
        default=str(_REPO / "harness/configs/agent_core_recovery.yaml"),
    )
    p.add_argument("--vllm-port", type=int, default=9300)
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p.parse_args()


def _apply_rollout_env() -> None:
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "rollback_events.jsonl"
    done_qids: set[str] = set()
    if args.resume and events_path.exists():
        for line in events_path.open(encoding="utf-8"):
            if line.strip():
                done_qids.add(str(json.loads(line).get("query_id", "")))

    _apply_rollout_env()
    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)

    shard_qids = _load_shard_queries(Path(args.manifest), args.shard, args.n_shards)
    records = [
        r for r in _load_records(Path(args.manifest), shard_qids)
        if r.query_id not in done_qids
    ]
    logger.info(
        "rollback collect shard=%s mode=%s pending_queries=%d",
        args.shard,
        args.mode,
        len(records),
    )
    if not records:
        logger.warning("no pending queries for rollback collection")
        return

    index_path = check_retrieval_backend("bm25", bm25_index_path=None)
    runtime = build_rollout_runtime(
        "browsecompplus",
        collection_split="test",
        reranker="none",
        retrieval="bm25",
        bm25_index_path=index_path,
    )

    vllm_handle = start_vllm_server(
        model_path=args.model_path,
        port=args.vllm_port,
        tensor_parallel_size=1,
        max_model_len=32768,
        served_model_name="rollback-collect",
        log_path=str(out_dir / "vllm_server.log"),
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    os.environ["base_url"] = vllm_handle.base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = "rollback-collect"
    get_llm_settings.cache_clear()

    client = get_llm_client()
    model_name = get_llm_model_name()
    inference = OpenAIAgentInferenceModel(
        openai_client=client,
        model=model_name,
        max_output_tokens=2048,
        temperature=1.0,
        api_style="chat_completions",
    )

    shadow = RollbackBilateralShadow()
    sem = asyncio.Semaphore(args.parallel)
    write_lock = asyncio.Lock()
    stats = {
        "n_events": 0,
        "n_rollback": 0,
        "n_continue": 0,
        "n_queries": 0,
        "n_errors": 0,
        "visibility_violation": 0,
        "schema_invalid": 0,
        "shadow_mutation": 0,
        "state_hash_mismatch": 0,
    }

    async def _one(record) -> None:
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
            store = CheckpointStore(branch_id=record.query_id)
            detector = StagnationDetector()
            driver = ChatDecisionDriver(env=env, inference=inference, max_turns=args.max_turns)
            query_rows: list[dict[str, Any]] = []

            def hook(state: DecisionState, action) -> Any:
                store.save_from_env(env, turn_id=int(state.turn_id))
                event = detector.observe_turn(env, checkpoint_store=store)
                if (
                    event is None
                    and args.mode == "injected"
                    and len(store.list_ids()) >= 2
                ):
                    event = FailureEvent(
                        RollbackReasonCode.QUERY_LOOP,
                        "injected",
                        suggested_checkpoint_id=store.list_ids()[0],
                    )
                if event is None:
                    label = shadow.label_failure_event(
                        FailureEvent(RollbackReasonCode.NONE, "healthy"),
                        healthy_continue=True,
                    )
                else:
                    label = shadow.label_failure_event(event)
                ds = build_rollback_decision_state(
                    state,
                    recent_queries=list(env.wm.search_history),
                    available_checkpoints=store.lightweight_metadata(),
                    remaining_search_budget=max(
                        0, args.max_turns - int(env._current_turn)
                    ),
                    remaining_recovery_budget=3,
                    branch_id=record.query_id,
                    state_hash=env.wm.snapshot_hash(),
                )
                query_rows.append(
                    {
                        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                        "query_id": record.query_id,
                        "turn_id": state.turn_id,
                        "decision_state": ds,
                        "operation": label.operation.value,
                        "checkpoint_id": label.checkpoint_id,
                        "reason_code": label.reason_code.value,
                        "route": label.route,
                        "mode": args.mode,
                    }
                )
                return action

            try:
                await asyncio.wait_for(driver.run(pre_step_hook=hook), timeout=600.0)
                stats["n_queries"] += 1
            except Exception as exc:
                stats["n_errors"] += 1
                logger.exception("rollback collect failed qid=%s: %s", record.query_id, exc)

            if not query_rows:
                logger.warning("no hook events for qid=%s", record.query_id)
                return

            async with write_lock:
                with events_path.open("a", encoding="utf-8") as f:
                    for row in query_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["n_events"] += 1
                        if row["operation"] == "ROLLBACK_TO":
                            stats["n_rollback"] += 1
                        else:
                            stats["n_continue"] += 1

    await asyncio.gather(*[_one(r) for r in records])
    vllm_handle.stop()
    (out_dir / "collection_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("collection done: %s", json.dumps(stats))
    print(json.dumps(stats, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
