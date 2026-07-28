#!/usr/bin/env python3
"""SCOPE Pre-training Shadow Audit on Bare BrowseComp+ trajectories.

Two modes:
  bare-replay  — reconstruct stop-state from bare answers; run M1/M2 shadows (CPU)
  online       — on-policy multi-turn search with same model; shadow at critical states

Both modes join Bare success/fail labels to compute:
  P(CORRECT | bare fail) and P(CORRECT | bare success)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.artifacts.schema import GuidanceMode
from harness.artifacts.visibility import mask_artifact_if_invalid
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.adapters import parse_action_from_tools
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import DecisionState, compute_text_hash
from harness.harness_config import apply_harness_config, config_path, load_harness_config
from harness.shadow.registry import build_default_registry
from inference.evaluate_harness1_vllm import VllmTokenCompleter
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server
from training.opd_v2.candidates import fill_recommended_action as fill_rec
from training.opd_v2.router import GuidanceRouter
from training.scope_config import load_scope_config, scope_section
from training.train_rl import MAX_TURNS, SlidingWindowSearchEnv
from harness.agent import TinkerAgentInferenceModel
from harness.tools import UserTextTool


def _normalize_answer(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def grade_bare_success(response_text: str, gold_answer: str) -> bool:
    """BrowseComp-style substring match after light normalization."""
    if not response_text or not gold_answer:
        return False
    resp = _normalize_answer(response_text)
    gold = _normalize_answer(gold_answer)
    if not gold:
        return False
    if gold in resp:
        return True
    # Also try raw casefold containment for short answers
    return gold_answer.strip().lower() in response_text.lower()


def load_bare_trajectories(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_gold_answers() -> dict[str, str]:
    answers: dict[str, str] = {}
    default = (
        _REPO_ROOT
        / "external"
        / "BrowseComp-Plus"
        / "data"
        / "browsecomp_plus_decrypted.jsonl"
    )
    path = Path(os.environ.get("BROWSECOMPPLUS_ANSWERS_PATH", str(default)))
    if not path.exists():
        return answers
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id", ""))
            if qid:
                answers[qid] = str(obj.get("answer", ""))
    return answers


@dataclass
class ModuleCounters:
    calls: int = 0
    endorse: int = 0
    correct: int = 0
    noop: int = 0
    masked: int = 0
    correct_verifier_pass: int = 0
    correct_verifier_total: int = 0

    def as_rates(self) -> dict[str, float]:
        n = max(self.calls, 1)
        return {
            "endorse_rate": self.endorse / n,
            "correct_rate": self.correct / n,
            "noop_rate": self.noop / n,
            "masked_rate": self.masked / n,
            "valid_correct_rate": (
                self.correct_verifier_pass / max(self.correct_verifier_total, 1)
            ),
        }


@dataclass
class AuditStats:
    num_states: int = 0
    num_endorse: int = 0
    num_correct: int = 0
    num_noop: int = 0
    num_masked: int = 0
    correct_verifier_pass: int = 0
    correct_verifier_total: int = 0
    artifact_leakage: int = 0
    artifact_total: int = 0
    by_module: dict[str, ModuleCounters] = field(
        default_factory=lambda: defaultdict(ModuleCounters)
    )
    correct_given_bare_fail: int = 0
    states_bare_fail: int = 0
    correct_given_bare_success: int = 0
    states_bare_success: int = 0
    episodes: int = 0
    bare_success_episodes: int = 0
    bare_fail_episodes: int = 0

    def record_decision(
        self,
        *,
        module_id: str,
        mode: GuidanceMode,
        masked: bool,
        verifier_pass: bool | None,
        leakage: bool,
        bare_success: bool,
    ) -> None:
        self.num_states += 1
        self.artifact_total += 1
        mc = self.by_module[module_id]
        mc.calls += 1
        if bare_success:
            self.states_bare_success += 1
        else:
            self.states_bare_fail += 1

        if leakage:
            self.artifact_leakage += 1
        if masked:
            self.num_masked += 1
            mc.masked += 1
            self.num_noop += 1
            mc.noop += 1
            return

        if mode == GuidanceMode.ENDORSE:
            self.num_endorse += 1
            mc.endorse += 1
        elif mode == GuidanceMode.CORRECT:
            self.num_correct += 1
            mc.correct += 1
            self.correct_verifier_total += 1
            mc.correct_verifier_total += 1
            if verifier_pass:
                self.correct_verifier_pass += 1
                mc.correct_verifier_pass += 1
            if bare_success:
                self.correct_given_bare_success += 1
            else:
                self.correct_given_bare_fail += 1
        else:
            self.num_noop += 1
            mc.noop += 1

    def to_dict(self) -> dict[str, Any]:
        n_states = max(self.num_states, 1)
        return {
            "num_states": self.num_states,
            "num_endorse": self.num_endorse,
            "num_correct": self.num_correct,
            "num_noop": self.num_noop,
            "num_masked": self.num_masked,
            "correct_verifier_pass_rate": (
                self.correct_verifier_pass / max(self.correct_verifier_total, 1)
            ),
            "artifact_leakage_rate": self.artifact_leakage / max(self.artifact_total, 1),
            "P_CORRECT_given_bare_fail": (
                self.correct_given_bare_fail / max(self.states_bare_fail, 1)
            ),
            "P_CORRECT_given_bare_success": (
                self.correct_given_bare_success / max(self.states_bare_success, 1)
            ),
            "endorse_rate": self.num_endorse / n_states,
            "correct_rate": self.num_correct / n_states,
            "noop_rate": self.num_noop / n_states,
            "episodes": self.episodes,
            "bare_success_episodes": self.bare_success_episodes,
            "bare_fail_episodes": self.bare_fail_episodes,
            "Evidence": self.by_module["evidence_state"].as_rates()
            | {"calls": self.by_module["evidence_state"].calls},
            "Verification": self.by_module["verification"].as_rates()
            | {"calls": self.by_module["verification"].calls},
        }


def _merge_stats(dst: AuditStats, src: AuditStats) -> None:
    dst.num_states += src.num_states
    dst.num_endorse += src.num_endorse
    dst.num_correct += src.num_correct
    dst.num_noop += src.num_noop
    dst.num_masked += src.num_masked
    dst.correct_verifier_pass += src.correct_verifier_pass
    dst.correct_verifier_total += src.correct_verifier_total
    dst.artifact_leakage += src.artifact_leakage
    dst.artifact_total += src.artifact_total
    dst.correct_given_bare_fail += src.correct_given_bare_fail
    dst.states_bare_fail += src.states_bare_fail
    dst.correct_given_bare_success += src.correct_given_bare_success
    dst.states_bare_success += src.states_bare_success
    dst.episodes += src.episodes
    dst.bare_success_episodes += src.bare_success_episodes
    dst.bare_fail_episodes += src.bare_fail_episodes
    for mid, mc in src.by_module.items():
        d = dst.by_module[mid]
        d.calls += mc.calls
        d.endorse += mc.endorse
        d.correct += mc.correct
        d.noop += mc.noop
        d.masked += mc.masked
        d.correct_verifier_pass += mc.correct_verifier_pass
        d.correct_verifier_total += mc.correct_verifier_total


def _build_selector(scope: dict[str, Any]) -> RuleBasedCriticalStateSelector:
    sel = scope.get("selector") or {}
    mods = scope.get("modules") or {}
    return RuleBasedCriticalStateSelector(
        SelectorConfig(
            before_stop=bool(sel.get("before_stop", True)),
            after_curate=bool(sel.get("after_curate", True)),
            after_verify=bool(sel.get("after_verify", True)),
            after_review=bool(sel.get("after_review", True)),
            after_pool_growth=bool(sel.get("after_pool_growth", True)),
            repeated_query=bool(sel.get("repeated_query", False)),
            low_remaining_turns=bool(sel.get("low_remaining_turns", False)),
            evidence_enabled=bool(mods.get("evidence_state", True)),
            verification_enabled=bool(mods.get("verification", True)),
            budget_enabled=bool(mods.get("budget_control", False)),
        )
    )


def audit_state_action(
    *,
    state: DecisionState,
    student_action: CapabilityAction,
    registry,
    selector: RuleBasedCriticalStateSelector,
    router: GuidanceRouter,
    stats: AuditStats,
    bare_success: bool,
    events_out: list[dict[str, Any]] | None = None,
) -> None:
    module_ids = selector.select(state, student_action)
    if not module_ids:
        # Still count as a critical-state miss / noop opportunity? Skip.
        return

    for mid in module_ids:
        if not registry.has(mid):
            continue
        module = registry.get(mid)
        artifact = module.analyze(state, student_action)
        if artifact.mode == GuidanceMode.CORRECT:
            artifact = fill_rec(state, artifact)
        artifact, vis = mask_artifact_if_invalid(state, artifact)
        masked = (not vis.valid) or bool(artifact.metadata.get("masked"))
        leakage = not vis.valid
        decision = router.route(state, artifact, module=module)

        verifier_pass = None
        if decision.mode == GuidanceMode.CORRECT:
            verifier_pass = bool(decision.validation.valid)

        stats.record_decision(
            module_id=mid,
            mode=decision.mode,
            masked=masked or decision.mode == GuidanceMode.IGNORE and leakage,
            verifier_pass=verifier_pass,
            leakage=leakage,
            bare_success=bare_success,
        )
        if events_out is not None:
            events_out.append(
                {
                    "episode_id": state.episode_id,
                    "task_id": state.task_id,
                    "turn_id": state.turn_id,
                    "module_id": mid,
                    "mode": decision.mode.value,
                    "reason_code": decision.artifact.reason_code,
                    "masked": masked,
                    "leakage": leakage,
                    "verifier_pass": verifier_pass,
                    "bare_success": bare_success,
                    "student_action": student_action.action_type.value,
                    "triggers": [
                        t.trigger
                        for t in selector.last_triggers
                        if t.module_id == mid
                    ],
                }
            )


def make_bare_stop_state(query_id: str, query: str, response_text: str) -> tuple[DecisionState, CapabilityAction]:
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": response_text[:500], "answer": response_text[:2000]},
    )
    state = DecisionState(
        episode_id=f"bare_{query_id}",
        task_id=str(query_id),
        turn_id=0,
        query=query,
        rendered_context=f"Query: {query}",
        action_history=(),
        observation_ids=(),
        visible_document_ids=(),
        pool_document_ids=(),
        curated_document_ids=(),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=0,
        remaining_search_calls=None,
        token_budget_used=0,
        token_budget_total=8192,
        last_action_type=None,
        repeated_query_score=0.0,
        wm_snapshot_hash=compute_text_hash(query),
    )
    return state, action


def run_bare_replay(
    *,
    bare_rows: list[dict[str, Any]],
    gold: dict[str, str],
    scope: dict[str, Any],
    out_dir: Path,
    limit: int = 0,
) -> dict[str, Any]:
    registry = build_default_registry(
        evidence_state=bool((scope.get("modules") or {}).get("evidence_state", True)),
        verification=bool((scope.get("modules") or {}).get("verification", True)),
        budget_control=False,
    )
    selector = _build_selector(scope)
    router = GuidanceRouter()
    stats = AuditStats()
    events: list[dict[str, Any]] = []

    rows = bare_rows[:limit] if limit > 0 else bare_rows
    for row in rows:
        qid = str(row["query_id"])
        query = str(row.get("query", ""))
        response = str(row.get("response_text", ""))
        gold_ans = gold.get(qid, "")
        bare_ok = grade_bare_success(response, gold_ans)
        stats.episodes += 1
        if bare_ok:
            stats.bare_success_episodes += 1
        else:
            stats.bare_fail_episodes += 1

        state, action = make_bare_stop_state(qid, query, response)
        audit_state_action(
            state=state,
            student_action=action,
            registry=registry,
            selector=selector,
            router=router,
            stats=stats,
            bare_success=bare_ok,
            events_out=events,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = stats.to_dict()
    summary["mode"] = "bare-replay"
    summary["n_bare_rows"] = len(rows)
    (out_dir / "shadow_audit_bare_replay_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    events_path = out_dir / "shadow_audit_bare_replay_events.jsonl"
    with events_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"[audit] bare-replay done: {json.dumps(summary, indent=2)}", flush=True)
    return summary


async def _audit_one_episode(
    *,
    qid: str,
    query: str,
    bare_success: bool,
    runtime,
    policy: VllmTokenCompleter,
    max_turns: int,
    registry,
    selector: RuleBasedCriticalStateSelector,
    router: GuidanceRouter,
    stats: AuditStats,
    events_out: list[dict[str, Any]],
) -> dict[str, Any]:
    env = SlidingWindowSearchEnv(
        toolset=runtime.toolset,
        search_tool=runtime.search_tool,
        query_id=qid,
        query_text=query,
        dataset=runtime.dataset,
        text_token_counter=runtime.text_token_counter,
        max_turns=max_turns,
    )
    ob, stop_condition = await env.initial_observation()
    turns = 0
    start = time.time()
    while True:
        # Export state BEFORE the student action executes (current WM).
        state_before = env.export_decision_state()
        ac_with_logprobs = await policy(ob, stop_condition)

        # Peek-parse action for shadow without mutating env; then step.
        full_toolset = env._build_full_toolset()
        try:
            action = TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
                env.enc, ac_with_logprobs.tokens, full_toolset
            )
            names: list[str] = []
            params_list: list[dict[str, Any]] = []
            for tool, params in zip(action.tools, action.params):
                if isinstance(tool, UserTextTool):
                    names.append("user_text")
                else:
                    names.append(tool.tool_schema.name)
                params_list.append(dict(params) if isinstance(params, dict) else {})
            cap = parse_action_from_tools(names, params_list)
        except Exception:
            cap = None

        if cap is not None:
            audit_state_action(
                state=state_before,
                student_action=cap,
                registry=registry,
                selector=selector,
                router=router,
                stats=stats,
                bare_success=bare_success,
                events_out=events_out,
            )

        step_result = await env.step(ac_with_logprobs.tokens)
        turns += 1
        if step_result.episode_done:
            break
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition

    stats.episodes += 1
    if bare_success:
        stats.bare_success_episodes += 1
    else:
        stats.bare_fail_episodes += 1

    return {
        "query_id": qid,
        "bare_success": bare_success,
        "turns": turns,
        "elapsed_s": round(time.time() - start, 1),
        "recall": float(env._terminal_metrics.get("recall", 0.0)),
        "final_answer_recall": float(
            env._terminal_metrics.get("final_answer_recall", 0.0)
        ),
        "error": env._terminal_metrics.get("no_error", 1.0) == 0.0,
    }


async def run_online_audit(
    *,
    bare_rows: list[dict[str, Any]],
    gold: dict[str, str],
    scope: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)
    harness_cfg.save_resolved(out_dir / "harness_resolved_config.yaml")

    index_path = check_retrieval_backend(
        args.retrieval,
        bm25_index_path=args.bm25_index_path,
        smoke=False,
    )
    runtime = build_rollout_runtime(
        "browsecompplus",
        collection_split="test",
        reranker=args.reranker,
        retrieval=args.retrieval,
        bm25_index_path=index_path,
    )

    registry = build_default_registry(
        evidence_state=bool((scope.get("modules") or {}).get("evidence_state", True)),
        verification=bool((scope.get("modules") or {}).get("verification", True)),
        budget_control=False,
    )
    selector = _build_selector(scope)
    router = GuidanceRouter()
    stats = AuditStats()
    events: list[dict[str, Any]] = []

    # Preserve bare query order; optional limit.
    rows = bare_rows[: args.limit] if args.limit > 0 else bare_rows

    episode_jsonl = out_dir / "shadow_audit_online_episodes.jsonl"
    done_ids = load_completed_query_ids(episode_jsonl) if args.resume else set()
    pending = [r for r in rows if str(r["query_id"]) not in done_ids]
    print(
        f"[audit] online pending {len(pending)}/{len(rows)} "
        f"(resume={args.resume}, parallel={args.parallel})",
        flush=True,
    )

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
    try:
        if args.vllm_url is None and args.manage_vllm:
            print(
                f"[audit] Starting vLLM TP={args.tensor_parallel_size} at {base_url} ...",
                flush=True,
            )
            vllm_handle = start_vllm_server(
                model_path=args.model_path,
                port=args.vllm_port,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                served_model_name=args.vllm_model_name,
                log_path=str(out_dir / "vllm_server.log"),
            )
            base_url = vllm_handle.base_url
            print(f"[audit] vLLM ready: {base_url}", flush=True)

        policy = VllmTokenCompleter(
            base_url=base_url,
            model=args.vllm_model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=900,
        )

        sem = asyncio.Semaphore(args.parallel)
        write_lock = asyncio.Lock()
        stats_lock = asyncio.Lock()
        completed = len(rows) - len(pending)
        async def _one(row: dict[str, Any]) -> None:
            nonlocal completed
            qid = str(row["query_id"])
            query = str(row.get("query", ""))
            if not query:
                try:
                    _, query = runtime.dataset.get_query_by_id(qid)
                except Exception:
                    pass
            gold_ans = gold.get(qid, "")
            bare_ok = grade_bare_success(str(row.get("response_text", "")), gold_ans)
            ep_events: list[dict[str, Any]] = []
            # Per-episode stats buffer to avoid races; merge under lock.
            ep_stats = AuditStats()
            async with sem:
                result = await _audit_one_episode(
                    qid=qid,
                    query=query,
                    bare_success=bare_ok,
                    runtime=runtime,
                    policy=policy,
                    max_turns=args.max_turns,
                    registry=registry,
                    selector=selector,
                    router=router,
                    stats=ep_stats,
                    events_out=ep_events,
                )
            async with stats_lock:
                _merge_stats(stats, ep_stats)
                events.extend(ep_events)
            async with write_lock:
                completed += 1
                with episode_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                if args.log_every > 0 and completed % args.log_every == 0:
                    print(
                        f"[audit] progress {completed}/{len(rows)} "
                        f"last={qid} bare_ok={bare_ok} turns={result.get('turns')} "
                        f"correct={stats.num_correct} endorse={stats.num_endorse}",
                        flush=True,
                    )

        await asyncio.gather(*[_one(r) for r in pending])

        # If resuming, recompute stats from events file would be incomplete;
        # we only accumulate for this run's pending. Reload full events if needed.
        summary = stats.to_dict()
        summary["mode"] = "online"
        summary["n_bare_rows"] = len(rows)
        summary["model_path"] = args.model_path
        summary["harness_config"] = args.harness_config
        summary["retrieval"] = args.retrieval
        summary["max_turns"] = args.max_turns
        summary["max_tokens"] = args.max_tokens
        summary["temperature"] = args.temperature
        (out_dir / "shadow_audit_online_stats.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (out_dir / "shadow_audit_online_events.jsonl").open(
            "a" if args.resume else "w", encoding="utf-8"
        ) as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print(f"[audit] online done: {json.dumps(summary, indent=2)}", flush=True)
        return summary
    finally:
        if vllm_handle is not None:
            print("[audit] Stopping vLLM ...", flush=True)
            vllm_handle.stop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCOPE shadow audit on Bare trajectories")
    p.add_argument(
        "--mode",
        choices=["bare-replay", "online", "both"],
        default="both",
        help="bare-replay=CPU stop-state audit; online=GPU multi-turn; both=run replay then online",
    )
    p.add_argument(
        "--bare-jsonl",
        default="/data/ppnm/BiSHOP/outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl",
    )
    p.add_argument(
        "--config",
        default="configs/scope/shadow_audit_m1_m2.yaml",
    )
    p.add_argument(
        "--harness-config",
        default=str(config_path("modules_full.yaml")),
        help="Runtime harness YAML for online student env (WM builders)",
    )
    p.add_argument("--output-dir", default="outputs/scope_shadow_audit_bare")
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--vllm-port", type=int, default=8772)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--vllm-model-name", default="scope-shadow-audit")
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--manage-vllm", action="store_true", default=True)
    p.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--retrieval", default="bm25", choices=["bm25", "chroma"])
    p.add_argument("--bm25-index-path", default=None)
    p.add_argument("--reranker", default="none", choices=["none", "baseten", "vllm"])
    p.add_argument("--log-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_scope_config(args.config)
    scope = scope_section(cfg)
    bare_path = Path(args.bare_jsonl)
    if not bare_path.exists():
        raise SystemExit(f"Bare jsonl not found: {bare_path}")

    bare_rows = load_bare_trajectories(bare_path)
    gold = load_gold_answers()
    print(
        f"[audit] loaded bare={len(bare_rows)} gold_answers={len(gold)} "
        f"mode={args.mode} out={out_dir}",
        flush=True,
    )

    manifest = {
        "bare_jsonl": str(bare_path),
        "n_bare": len(bare_rows),
        "scope_config": args.config,
        "harness_config": args.harness_config,
        "model_path": args.model_path,
        "modes": [],
    }

    if args.mode in {"bare-replay", "both"}:
        run_bare_replay(
            bare_rows=bare_rows,
            gold=gold,
            scope=scope,
            out_dir=out_dir,
            limit=args.limit,
        )
        manifest["modes"].append("bare-replay")

    if args.mode in {"online", "both"}:
        asyncio.run(
            run_online_audit(
                bare_rows=bare_rows,
                gold=gold,
                scope=scope,
                out_dir=out_dir,
                args=args,
            )
        )
        manifest["modes"].append("online")

    (out_dir / "shadow_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[audit] All done -> {out_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-dummy"))
    main()
