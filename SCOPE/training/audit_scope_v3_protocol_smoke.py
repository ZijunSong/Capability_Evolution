#!/usr/bin/env python3
"""SCOPE v3 protocol smoke: BrowseComp+ online ChatDecision audit with verified routing.

Outputs (under --output-dir):
  events.jsonl
  samples.jsonl
  summary.json
  resolved_config.yaml
  errors.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from harness.agent import OpenAIAgentInferenceModel
from harness.artifacts.gates import capture_env_fingerprint
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import ROUND1_ENABLED_CAPABILITIES
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.harness_config import apply_harness_config, config_path, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name
from harness.shadow.registry import build_default_registry
from harness.telemetry.writer import ScopeTelemetryWriter
from training.audit_scope_chat_online import _selector
from training.chat_decision_driver import ChatDecisionDriver, ChatTurnRecord
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server
from training.scope.audit_analytics import build_formal_audit_report, enrich_event_local_gt
from training.scope.pipeline import run_supervision_pipeline
from training.scope.schema import Route
from training.scope_config import load_scope_config, scope_section
from training.train_rl import MAX_TURNS, SlidingWindowSearchEnv


@dataclass
class CapRouteStats:
    calls: int = 0
    endorse: int = 0
    correct: int = 0
    ignore: int = 0

    def record(self, route: Route) -> None:
        self.calls += 1
        if route == Route.ENDORSE:
            self.endorse += 1
        elif route == Route.CORRECT:
            self.correct += 1
        else:
            self.ignore += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "endorse": self.endorse,
            "correct": self.correct,
            "ignore": self.ignore,
        }


@dataclass
class V3SmokeStats:
    n_queries: int = 0
    n_events: int = 0
    n_shadow_calls: int = 0
    duplicate_evidence: CapRouteStats = field(default_factory=CapRouteStats)
    premature_stop: CapRouteStats = field(default_factory=CapRouteStats)
    visibility_violations: int = 0
    shadow_mutations: int = 0
    invalid_actions: int = 0
    verifier_rejects: int = 0
    n_trainable_samples: int = 0
    n_endorse: int = 0
    n_correct: int = 0

    def record_pipeline(self, *, capability_id: str, route: Route, gates, sample) -> None:
        self.n_shadow_calls += 1
        bucket = {
            "duplicate_evidence": self.duplicate_evidence,
            "premature_stop": self.premature_stop,
        }.get(capability_id)
        if bucket is not None:
            bucket.record(route)
        if not gates.visible:
            self.visibility_violations += 1
        if not gates.purity_ok:
            self.shadow_mutations += 1
        if not gates.executable:
            self.invalid_actions += 1
        if sample.verification.target_valid is False:
            self.verifier_rejects += 1
        if sample.train_mask:
            self.n_trainable_samples += 1
        if route == Route.ENDORSE:
            self.n_endorse += 1
        elif route == Route.CORRECT:
            self.n_correct += 1

    def to_summary(self) -> dict[str, Any]:
        n = max(1, self.n_shadow_calls)
        train_routes = max(1, self.n_endorse + self.n_correct)
        return {
            "mode": "scope_v3_protocol_smoke",
            "n_queries": self.n_queries,
            "n_events": self.n_events,
            "n_shadow_calls": self.n_shadow_calls,
            "Dup": self.duplicate_evidence.to_dict(),
            "Premature": self.premature_stop.to_dict(),
            "visibility_violation_rate": self.visibility_violations / n,
            "shadow_mutation_rate": self.shadow_mutations / n,
            "invalid_action_rate": self.invalid_actions / n,
            "verifier_reject_rate": self.verifier_rejects / n,
            "n_trainable_samples": self.n_trainable_samples,
            "endorse_correct_ratio": self.n_endorse / train_routes,
            "n_endorse": self.n_endorse,
            "n_correct": self.n_correct,
            "n_ignore": self.n_shadow_calls - self.n_endorse - self.n_correct,
            "capabilities_enabled": [c.value for c in ROUND1_ENABLED_CAPABILITIES],
        }


def audit_turn_v3(
    turn: ChatTurnRecord,
    *,
    env: SlidingWindowSearchEnv,
    registry,
    selector: RuleBasedCriticalStateSelector,
    stats: V3SmokeStats,
    telemetry: ScopeTelemetryWriter,
    events: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    enabled_caps: frozenset[str],
) -> None:
    state = turn.decision_state
    action = turn.student_action
    module_ids = selector.select(state, action)
    for mid in module_ids:
        if not registry.has(mid):
            continue
        module = registry.get(mid)
        fp_before = capture_env_fingerprint(env)
        try:
            artifact = module.analyze(state, action)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "kind": "shadow_analyze_error",
                    "episode_id": state.episode_id,
                    "turn_id": state.turn_id,
                    "module_id": mid,
                    "error": str(exc)[:500],
                }
            )
            continue
        fp_after = capture_env_fingerprint(env)
        result = run_supervision_pipeline(
            state,
            action,
            artifact=artifact,
            fingerprint_before=fp_before,
            fingerprint_after=fp_after,
            event_id=f"{state.episode_id}:{state.turn_id}:{mid}",
            student_state_text=state.rendered_context,
            telemetry=telemetry,
            enforce_round1_capability_filter=True,
        )
        cap = result.artifact.resolved_capability().value
        if cap not in enabled_caps:
            # Still count shadow invocation in telemetry-only path below
            continue

        stats.record_pipeline(
            capability_id=cap,
            route=result.routing.route,
            gates=result.routing.gates,
            sample=result.sample,
        )
        ev = enrich_event_local_gt(
            {
            "event": "supervision_sample_emitted",
            "episode_id": state.episode_id,
            "turn_id": state.turn_id,
            "event_id": f"{state.episode_id}:{state.turn_id}:{mid}",
            "task_id": state.task_id,
            "module_id": mid,
            "capability_id": cap,
            "decision_state_hash": state.core_state_hash(),
            "decision_state": state.to_dict(),
            "student_action_struct": action.to_dict(),
            "student_action": action.to_dict(),
            "artifact": result.artifact.to_dict(),
            "gate_results": result.routing.gates.to_dict(),
            "candidate_action": (
                result.routing.candidate.action.to_dict()
                if result.routing.candidate
                else None
            ),
            "target_action": (
                result.routing.target_action.to_dict()
                if result.routing.target_action
                else None
            ),
            "verifier_result": result.sample.verification.to_dict(),
            "route": result.routing.route.value,
            "train_mask": result.sample.train_mask,
            "audit_error": result.sample.audit_error,
            "fingerprint_before": fp_before,
            "fingerprint_after": fp_after,
            }
        )
        events.append(ev)
        samples.append(result.sample.to_dict())
        if result.sample.audit_error:
            errors.append(
                {
                    "kind": "routing_audit_error",
                    "episode_id": state.episode_id,
                    "turn_id": state.turn_id,
                    "capability_id": cap,
                    "audit_error": result.sample.audit_error,
                    "route": result.routing.route.value,
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/scope/sdi_dup_premature.yaml")
    p.add_argument(
        "--harness-config",
        default=str(config_path("modules_full_v2.yaml")),
    )
    p.add_argument("--output-dir", default="outputs/scope_v3_protocol_smoke20")
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--vllm-port", type=int, default=8774)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--vllm-model-name", default="scope-v3-smoke")
    p.add_argument("--manage-vllm", action="store_true", default=False)
    p.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--retrieval", default="bm25", choices=["bm25", "chroma"])
    p.add_argument("--bm25-index-path", default=None)
    p.add_argument("--reranker", default="none")
    p.add_argument("--log-every", type=int, default=2)
    p.add_argument("--split", default="all")
    p.add_argument("--query-timeout-s", type=float, default=600.0)
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scope_cfg_raw = load_scope_config(args.config)
    scope = scope_section(scope_cfg_raw)
    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)

    resolved = {
        "scope": scope_cfg_raw.get("scope") or scope,
        "harness_config": str(args.harness_config),
        "model_path": args.model_path,
        "limit": args.limit,
        "seed": args.seed,
        "protocol": "scope_v3",
    }
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    harness_cfg.save_resolved(out_dir / "harness_resolved_config.yaml")

    enabled_caps = frozenset(
        c.value
        for c in ROUND1_ENABLED_CAPABILITIES
    )
    cap_cfg = (scope.get("capabilities") or {}).get("enabled")
    if cap_cfg:
        enabled_caps = frozenset(str(x) for x in cap_cfg)

    index_path = check_retrieval_backend(
        args.retrieval, bm25_index_path=args.bm25_index_path, smoke=False
    )
    runtime = build_rollout_runtime(
        "browsecompplus",
        collection_split="test",
        reranker=args.reranker,
        retrieval=args.retrieval,
        bm25_index_path=index_path,
    )

    records = load_browsecomp_full_queries(
        split=args.split, limit=0, download_if_missing=False
    )

    def _sample_key(qid: str) -> str:
        return hashlib.md5(f"{args.seed}:{qid}".encode()).hexdigest()

    records = sorted(records, key=lambda r: _sample_key(r.query_id))
    if args.limit > 0:
        records = records[: args.limit]

    events_path = out_dir / "events.jsonl"
    samples_path = out_dir / "samples.jsonl"
    errors_path = out_dir / "errors.jsonl"
    episodes_path = out_dir / "episodes.jsonl"
    done = load_completed_query_ids(episodes_path) if args.resume else set()
    pending = [r for r in records if r.query_id not in done]
    if not args.resume:
        for p in (events_path, samples_path, errors_path, episodes_path):
            p.write_text("", encoding="utf-8")
        done = set()
        pending = list(records)

    registry = build_default_registry(
        evidence_state=bool((scope.get("modules") or {}).get("evidence_state", True)),
        verification=bool((scope.get("modules") or {}).get("verification", True)),
        budget_control=False,
    )
    selector = _selector(scope)
    stats = V3SmokeStats(n_queries=len(records))
    telemetry = ScopeTelemetryWriter(out_dir / "telemetry_events.jsonl")

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
    os.environ["base_url"] = base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = args.vllm_model_name
    from harness.llm_env import get_llm_settings

    get_llm_settings.cache_clear()

    print(
        f"[v3-smoke] n_queries={len(records)} pending={len(pending)} out={out_dir}",
        flush=True,
    )

    try:
        if args.manage_vllm and args.vllm_url is None:
            vllm_handle = start_vllm_server(
                model_path=args.model_path,
                port=args.vllm_port,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                served_model_name=args.vllm_model_name,
                log_path=str(out_dir / "vllm_server.log"),
            )
            base_url = vllm_handle.base_url
            os.environ["base_url"] = base_url
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

        sem = asyncio.Semaphore(args.parallel)
        write_lock = asyncio.Lock()
        completed = len(records) - len(pending)

        async def _one(record) -> None:
            nonlocal completed
            ep_events: list[dict[str, Any]] = []
            ep_samples: list[dict[str, Any]] = []
            ep_errors: list[dict[str, Any]] = []
            result: dict[str, Any]
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
                    driver = ChatDecisionDriver(
                        env=env, inference=inference, max_turns=args.max_turns
                    )

                    def _on_turn(turn: ChatTurnRecord) -> None:
                        audit_turn_v3(
                            turn,
                            env=env,
                            registry=registry,
                            selector=selector,
                            stats=stats,
                            telemetry=telemetry,
                            events=ep_events,
                            samples=ep_samples,
                            errors=ep_errors,
                            enabled_caps=enabled_caps,
                        )

                    result = await asyncio.wait_for(
                        driver.run(on_critical_turn=_on_turn),
                        timeout=float(args.query_timeout_s),
                    )
                    turn_records = list(result.get("turn_records") or [])
                    last = turn_records[-1] if turn_records else None
                    last_atype = (
                        last.student_action.action_type if last is not None else None
                    )
                    if last is not None and last_atype not in {
                        CapabilityActionType.STOP_AND_ANSWER,
                        CapabilityActionType.ANSWER,
                        CapabilityActionType.ABSTAIN,
                    }:
                        terminal_state = env.export_decision_state()
                        terminal_turn = ChatTurnRecord(
                            turn_id=terminal_state.turn_id,
                            decision_state=terminal_state,
                            student_action=CapabilityAction(
                                action_type=CapabilityActionType.STOP_AND_ANSWER,
                                arguments={
                                    "reasoning": "episode_end_audit",
                                    "synthetic_terminal": True,
                                },
                            ),
                            action=last.action,
                            observation_text="synthetic_terminal_stop",
                            episode_done=True,
                            metrics=dict(result.get("metrics") or {}),
                        )
                        audit_turn_v3(
                            terminal_turn,
                            env=env,
                            registry=registry,
                            selector=selector,
                            stats=stats,
                            telemetry=telemetry,
                            events=ep_events,
                            samples=ep_samples,
                            errors=ep_errors,
                            enabled_caps=enabled_caps,
                        )
            except asyncio.TimeoutError:
                ep_errors.append(
                    {
                        "kind": "query_timeout",
                        "query_id": record.query_id,
                        "error": f"timeout_{args.query_timeout_s}s",
                    }
                )
                result = {"query_id": record.query_id, "error": True, "turns": 0}
            except Exception as exc:  # noqa: BLE001
                ep_errors.append(
                    {
                        "kind": "query_error",
                        "query_id": record.query_id,
                        "error": str(exc)[:500],
                    }
                )
                result = {"query_id": record.query_id, "error": True, "turns": 0}

            row = {
                "query_id": record.query_id,
                "turns": result.get("turns", 0),
                "recall": result.get("recall", 0.0),
                "n_shadow_events": len(ep_events),
                "error": result.get("error", False),
            }
            async with write_lock:
                completed += 1
                with episodes_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if ep_events:
                    with events_path.open("a", encoding="utf-8") as fh:
                        for ev in ep_events:
                            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                if ep_samples:
                    with samples_path.open("a", encoding="utf-8") as fh:
                        for s in ep_samples:
                            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                if ep_errors:
                    with errors_path.open("a", encoding="utf-8") as fh:
                        for e in ep_errors:
                            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
                if args.log_every and completed % args.log_every == 0:
                    print(
                        f"[v3-smoke] {completed}/{len(records)} "
                        f"last={record.query_id} events={len(ep_events)}",
                        flush=True,
                    )

        await asyncio.gather(*[_one(r) for r in pending])

        # Recompute from disk for resume safety
        all_events: list[dict[str, Any]] = []
        all_samples: list[dict[str, Any]] = []
        if events_path.exists():
            with events_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        all_events.append(json.loads(line))
        if samples_path.exists():
            with samples_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        all_samples.append(json.loads(line))

        final_stats = V3SmokeStats(n_queries=len(records))
        final_stats.n_events = len(all_events)
        for ev in all_events:
            cap = str(ev.get("capability_id", ""))
            route = Route(str(ev.get("route", "IGNORE")).upper())
            gates_raw = ev.get("gate_results") or {}
            class _G:
                visible = bool(gates_raw.get("visible", True))
                purity_ok = bool(gates_raw.get("purity_ok", True))
                executable = bool(gates_raw.get("executable", True))
            ver = ev.get("verifier_result") or {}
            class _S:
                train_mask = int(ev.get("train_mask", 0))
                verification = type("V", (), {"target_valid": ver.get("target_valid")})()
            final_stats.record_pipeline(
                capability_id=cap, route=route, gates=_G(), sample=_S()
            )

        summary = final_stats.to_summary()
        summary["telemetry_stats"] = telemetry.flush_stats()
        summary["n_samples"] = len(all_samples)
        summary["model_path"] = args.model_path
        summary["config"] = str(args.config)
        summary["harness_config"] = str(args.harness_config)
        summary = build_formal_audit_report(
            all_events,
            n_queries=len(records),
            base_summary=summary,
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if vllm_handle is not None:
            vllm_handle.stop()


def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
