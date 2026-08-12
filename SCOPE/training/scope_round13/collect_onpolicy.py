#!/usr/bin/env python3
"""Round13 Barrier1: one-pass on-policy same-state shadow distillation collection.

Student (Round11 full_stage1 A0) produces trajectories; shadow labels the same
pre-action state; ROLLBACK_TO executes via canonical pick_rollback_checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from harness.agent import OpenAIAgentInferenceModel
from harness.capability.rollback_operation import RollbackOperation, RollbackReasonCode
from harness.capability.state import DecisionState
from harness.harness_config import apply_harness_config, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name, get_llm_settings
from harness.recovery.checkpoint_store import CheckpointStore
from harness.recovery.recovery_budget import RecoveryBudget
from harness.recovery.rollback_runtime import RollbackRuntime
from harness.recovery.stagnation_detector import FailureEvent, StagnationDetector
from training.chat_decision_driver import ChatDecisionDriver
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.vllm_server import start_vllm_server
from training.scope.decide_rollback_operation import RollbackDecision, decide_rollback_operation
from training.scope.rollback_action_realizer import RollbackActionRealizer
from training.scope.rollback_decision_state import build_rollback_decision_state
from training.scope.rollback_operation_objectives import format_rollback_operation_prompt
from training.scope.rollback_operation_runtime import pick_rollback_checkpoint
from training.scope.rollback_shadow import RollbackBilateralShadow
from training.scope.vllm_rollback_scorer import VllmRollbackScorer
from training.scope_round8.agent_core_rollout import _load_records
from training.scope_round3.hmin_v2_dup_rollout import _load_shard_queries
from training.train_rl import SlidingWindowSearchEnv


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard", default="shard0")
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument(
        "--model-path",
        default=str(
            _REPO
            / "outputs/scope_round11/phase_b/factorized_full_stage1_seed42/merged"
        ),
    )
    p.add_argument(
        "--harness-config",
        default=str(_REPO / "harness/configs/agent_core_recovery.yaml"),
    )
    p.add_argument("--vllm-port", type=int, default=18700)
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=35)
    p.add_argument("--query-timeout-s", type=float, default=600.0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--split-name", default="train")
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


def _visible_features(ds: dict[str, Any]) -> dict[str, Any]:
    cands = list(ds.get("available_checkpoints") or [])
    ages = []
    turn = int(ds.get("turn_id", 0))
    for c in cands:
        ages.append(max(0, turn - int(c.get("turn_id", 0))))
    return {
        "turn": turn,
        "candidate_count": len(cands),
        "rollback_budget_remaining": ds.get("remaining_recovery_budget"),
        "latest_checkpoint_age": min(ages) if ages else None,
        "successful_checkpoint_count": sum(
            1 for c in cands if int(c.get("n_verified", c.get("verified_count", 0)) or 0) > 0
        ),
        "failure_count": int(ds.get("repeated_query_count") or 0),
        "progress_since_checkpoint": ds.get("progress_since_checkpoint"),
        "new_evidence_since_checkpoint": ds.get("new_evidence_since_checkpoint"),
        "remaining_search_budget": ds.get("remaining_search_budget"),
    }


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_env()
    hb_path = out_dir / "HEARTBEAT"
    hb_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")

    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)

    shard_qids = _load_shard_queries(Path(args.manifest), args.shard, args.n_shards)
    records = _load_records(Path(args.manifest), shard_qids)

    events_path = out_dir / "rollback_events.jsonl"
    episodes_path = out_dir / "episodes.jsonl"
    done = load_completed_query_ids(episodes_path) if args.resume else set()
    # Also skip queries already present in events (partial resume)
    if args.resume and events_path.exists():
        for line in events_path.open(encoding="utf-8"):
            if line.strip():
                done.add(str(json.loads(line).get("query_id", "")))
    pending = [r for r in records if r.query_id not in done]

    resolved = {
        "schema_version": "scope.round13.onpolicy_collect.v1",
        "split_name": args.split_name,
        "model_path": args.model_path,
        "manifest": str(args.manifest),
        "shard": args.shard,
        "n_shards": args.n_shards,
        "n_pending": len(pending),
        "n_done": len(done),
        "tau": 0.0,
        "disable_replan": True,
        "checkpoint_resolver": "canonical_pick_rollback_checkpoint",
        "stage1_view": "A0",
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    if not pending:
        (out_dir / "collection_stats.json").write_text(
            json.dumps({"n_queries": 0, "skipped_all_done": True}, indent=2) + "\n"
        )
        (out_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
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
        served_model_name="r13-onpolicy-collect",
        log_path=str(out_dir / "vllm_server.log"),
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    os.environ["base_url"] = vllm_handle.base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = "r13-onpolicy-collect"
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
    scorer = VllmRollbackScorer(
        client=client, model=model_name, model_path=args.model_path, hint=""
    )
    shadow = RollbackBilateralShadow()
    realizer = RollbackActionRealizer()
    detector = StagnationDetector()

    sem = asyncio.Semaphore(args.parallel)
    write_lock = asyncio.Lock()
    stats = {
        "n_queries": 0,
        "n_errors": 0,
        "n_events": 0,
        "n_continue_gold": 0,
        "n_rollback_gold": 0,
        "n_student_continue": 0,
        "n_student_rollback": 0,
    }

    async def _one(record) -> None:
        async with sem:
            hb_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")
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
            budget = RecoveryBudget(max_rollbacks=3)
            rb_runtime = RollbackRuntime(store, budget)
            query_rows: list[dict[str, Any]] = []
            ep: dict[str, Any] = {"query_id": record.query_id, "error": False}

            def hook(state: DecisionState, action: Any) -> Any:
                store.save_from_env(env, turn_id=int(state.turn_id))
                failure = detector.observe_turn(env, checkpoint_store=store)
                if failure is None:
                    shadow_label = shadow.label_failure_event(
                        FailureEvent(RollbackReasonCode.NONE, "healthy"),
                        healthy_continue=True,
                    )
                    suggested = None
                else:
                    shadow_label = shadow.label_failure_event(failure)
                    suggested = failure.suggested_checkpoint_id

                ds = build_rollback_decision_state(
                    state,
                    recent_queries=list(env.wm.search_history),
                    available_checkpoints=store.lightweight_metadata(),
                    remaining_search_budget=max(
                        0, args.max_turns - int(env._current_turn)
                    ),
                    remaining_recovery_budget=budget.remaining(),
                    branch_id=record.query_id,
                    state_hash=env.wm.snapshot_hash(),
                )
                state_text = str(
                    ds.get("rendered_context") or ds.get("student_state_text") or ""
                )
                a0_prompt = format_rollback_operation_prompt(
                    state_text,
                    available_checkpoints=list(ds.get("available_checkpoints") or []),
                    hint="",
                )
                score_res = scorer.score_final_prompt(a0_prompt)
                ck_meta = list(ds.get("available_checkpoints") or [])
                ck_pick = pick_rollback_checkpoint(
                    ck_meta, int(ds.get("turn_id", 0)), suggested=suggested
                )
                decision = decide_rollback_operation(
                    score_continue=score_res.scores[RollbackOperation.CONTINUE.value],
                    score_replan=score_res.scores[RollbackOperation.REPLAN.value],
                    score_rollback=score_res.scores[RollbackOperation.ROLLBACK_TO.value],
                    threshold=0.0,
                    candidate_checkpoint_id=ck_pick,
                    disable_replan=True,
                )
                pred_op = decision.predicted_operation
                pred_ck = decision.checkpoint_id
                resolver_reason = "canonical_latest_eligible"
                if suggested and pred_ck == suggested:
                    resolver_reason = "shadow_suggested_if_eligible_else_latest"

                valid_ids = {str(c.get("checkpoint_id", "")) for c in ck_meta}
                if pred_op == RollbackOperation.ROLLBACK_TO and pred_ck not in valid_ids:
                    pred_op = RollbackOperation.CONTINUE
                    pred_ck = None
                    resolver_reason = "invalid_checkpoint_fallback_continue"
                if pred_op == RollbackOperation.ROLLBACK_TO and not budget.can_rollback():
                    pred_op = RollbackOperation.CONTINUE
                    pred_ck = None
                    resolver_reason = "budget_exhausted_fallback_continue"

                s_c = float(score_res.scores[RollbackOperation.CONTINUE.value])
                s_r = float(score_res.scores[RollbackOperation.ROLLBACK_TO.value])
                margin = s_r - s_c

                rb_decision = RollbackDecision(
                    predicted_operation=pred_op,
                    checkpoint_id=pred_ck
                    if pred_op == RollbackOperation.ROLLBACK_TO
                    else None,
                )
                ok = realizer.realize(env, rb_decision, rb_runtime)

                query_rows.append(
                    {
                        "event_id": f"r13_{uuid.uuid4().hex[:12]}",
                        "query_id": record.query_id,
                        "turn": int(state.turn_id),
                        "turn_id": int(state.turn_id),
                        "effective_state_hash": str(
                            ds.get("current_state_hash") or env.wm.snapshot_hash()
                        ),
                        "A0_prompt_hash": _sha256(a0_prompt),
                        "A0_prompt": a0_prompt,
                        "candidate_ids": [
                            str(c.get("checkpoint_id")) for c in ck_meta
                        ],
                        "student_operation": pred_op.value,
                        "student_margin": margin,
                        "student_scores": dict(score_res.scores),
                        "gold_operation": shadow_label.operation.value,
                        "gold_checkpoint_id": shadow_label.checkpoint_id,
                        "shadow_reason_code": shadow_label.reason_code.value,
                        "shadow_route": shadow_label.route,
                        "executed_checkpoint": pred_ck
                        if pred_op == RollbackOperation.ROLLBACK_TO
                        else None,
                        "checkpoint_resolver_reason": resolver_reason,
                        "rollback_realize_ok": bool(ok),
                        "student_visible_features": _visible_features(ds),
                        "decision_state": ds,
                        "student_state_text": state_text,
                        "operation": shadow_label.operation.value,
                        "gold_operation_alias": shadow_label.operation.value,
                        "target_action": {
                            "operation": shadow_label.operation.value,
                            "checkpoint_id": shadow_label.checkpoint_id,
                        },
                        "capability_id": "rollback",
                        "route": shadow_label.route,
                        "split": args.split_name,
                        "collection_policy": "r11_full_stage1_a0_seed42",
                    }
                )
                return action

            try:
                result = await asyncio.wait_for(
                    ChatDecisionDriver(
                        env=env, inference=inference, max_turns=args.max_turns
                    ).run(pre_step_hook=hook),
                    timeout=float(args.query_timeout_s),
                )
                ep.update({k: result.get(k) for k in ("turns", "recall", "reward") if k in result})
                stats["n_queries"] += 1
            except Exception as exc:
                stats["n_errors"] += 1
                ep["error"] = True
                ep["error_message"] = str(exc)

            async with write_lock:
                with episodes_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ep, ensure_ascii=False) + "\n")
                if query_rows:
                    with events_path.open("a", encoding="utf-8") as f:
                        for row in query_rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stats["n_events"] += 1
                            if row["gold_operation"] == "ROLLBACK_TO":
                                stats["n_rollback_gold"] += 1
                            else:
                                stats["n_continue_gold"] += 1
                            if row["student_operation"] == "ROLLBACK_TO":
                                stats["n_student_rollback"] += 1
                            else:
                                stats["n_student_continue"] += 1
                hb_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")

    await asyncio.gather(*[_one(r) for r in pending])
    vllm_handle.stop()
    (out_dir / "collection_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(json.dumps(stats, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
