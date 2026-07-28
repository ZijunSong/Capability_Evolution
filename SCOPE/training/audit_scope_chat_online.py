#!/usr/bin/env python3
"""SCOPE chat-online DecisionState audit (Qwen / OpenAI messages, no Harmony).

Sanity check on 20–50 BrowseComp queries:
  1) Drive SlidingWindowSearchEnv via chat completions
  2) Export DecisionState each turn (protocol-agnostic)
  3) Run M1 Evidence + M2 Verification shadows
  4) Local decision audit: Good / Bad / Ambiguous + intervention P/R
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.agent import OpenAIAgentInferenceModel
from harness.artifacts.schema import GuidanceMode
from harness.artifacts.visibility import mask_artifact_if_invalid
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import DecisionState
from harness.harness_config import apply_harness_config, config_path, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name
from harness.shadow.registry import build_default_registry
from training.chat_decision_driver import ChatDecisionDriver, ChatTurnRecord
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server
from training.opd_v2.candidates import fill_recommended_action
from training.opd_v2.router import GuidanceRouter
from training.scope_config import load_scope_config, scope_section
from training.train_rl import MAX_TURNS, SlidingWindowSearchEnv


LocalLabel = Literal["good", "bad", "ambiguous"]


def label_local_capabilities(
    state: DecisionState,
    action: CapabilityAction,
    module_id: str,
) -> list[str]:
    """Typed local GT capability tags (deterministic heuristics)."""
    caps: list[str] = []
    stopping = action.action_type in {
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.ABSTAIN,
    }
    curated = set(state.curated_document_ids)
    pool = set(state.pool_document_ids) | set(state.visible_document_ids)

    if module_id == "evidence_state":
        add_ids = action.arguments.get("add_ids") or []
        if not isinstance(add_ids, list):
            add_ids = []
        if add_ids and all(str(d) in curated for d in add_ids):
            caps.append("DUPLICATE_EVIDENCE")
        if any(
            str(c.status).lower() in {"conflict", "conflicting", "contradicted"}
            for c in state.evidence_claims
        ):
            caps.append("CONFLICTING_EVIDENCE")
        for rec in state.verification_records:
            vals = list(rec.judgments.values())
            if vals and (True in vals) and (False in vals):
                caps.append("CONFLICTING_EVIDENCE")
                break
        if any(not c.supporting_document_ids for c in state.evidence_claims):
            caps.append("MISSING_DIRECT_SUPPORT")
        for c in state.evidence_claims:
            if c.supporting_document_ids and not any(
                d in pool for d in c.supporting_document_ids
            ):
                caps.append("WRONG_CLAIM_BINDING")
                break
        status = str(action.arguments.get("status", "")).lower()
        if status in {"supported", "verified"} and not state.verification_records:
            caps.append("WEAK_SUPPORT")
        elif any(
            c.supporting_document_ids
            and c.status.lower() in {"unsupported", "weak", "unverified", "partial"}
            for c in state.evidence_claims
        ):
            caps.append("WEAK_SUPPORT")
        # Align typed GT with EvidenceShadow IRRELEVANT heuristic.
        from harness.shadow.evidence_shadow import _irrelevant_evidence

        if _irrelevant_evidence(state, add_ids) and action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
            CapabilityActionType.REVIEW_DOCS,
        }:
            caps.append("IRRELEVANT_EVIDENCE")

    if module_id == "verification":
        conflict = False
        for rec in state.verification_records:
            vals = list(rec.judgments.values())
            if vals and (True in vals) and (False in vals):
                conflict = True
                break
        if conflict:
            caps.append("UNRESOLVED_CONFLICT")
        missing = (not curated) or any(
            not c.supporting_document_ids for c in state.evidence_claims
        )
        # Positive verify evidence: at least one record with a True judgment.
        has_positive_verify = any(
            any(r.judgments.values()) for r in state.verification_records if r.judgments
        )
        unverified = (not state.verification_records) or (
            not has_positive_verify
            and any(
                r.judgments and not any(r.judgments.values())
                for r in state.verification_records
            )
        )
        if stopping and missing:
            caps.append("MISSING_DIRECT_EVIDENCE")
        elif stopping and unverified:
            caps.append("PREMATURE_STOP")
        # else: stopping with positive verify → no premature GT (valid stop)
        if action.action_type == CapabilityActionType.VERIFY_CLAIM:
            docs = action.arguments.get("doc_ids") or []
            visible = set(state.visible_document_ids) | set(state.pool_document_ids)
            if docs and any(str(d) not in visible for d in docs):
                caps.append("SOURCE_NOT_VISIBLE")
            if not docs or not str(action.arguments.get("claim", "")).strip():
                caps.append("MISSING_DIRECT_EVIDENCE")

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def label_local_decision(
    state: DecisionState,
    action: CapabilityAction,
    module_id: str,
) -> LocalLabel:
    """Heuristic local decision quality (not trajectory success).

    Typed capability tags that indicate a clear local error force ``bad``.
    """
    caps = label_local_capabilities(state, action, module_id)
    error_caps = {
        "DUPLICATE_EVIDENCE",
        "IRRELEVANT_EVIDENCE",
        "WEAK_SUPPORT",
        "MISSING_DIRECT_SUPPORT",
        "CONFLICTING_EVIDENCE",
        "WRONG_CLAIM_BINDING",
        "MISSING_DIRECT_EVIDENCE",
        "PREMATURE_STOP",
        "UNRESOLVED_CONFLICT",
        "INVALID_CITATION",
        "SOURCE_NOT_VISIBLE",
    }
    if any(c in error_caps for c in caps):
        return "bad"

    stopping = action.action_type in {
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.ABSTAIN,
    }
    curated = set(state.curated_document_ids)
    pool = set(state.pool_document_ids)
    visible = set(state.visible_document_ids) | pool

    if module_id == "verification":
        if stopping:
            # Reach here only when typed GT did not mark premature/missing.
            has_positive = any(
                any(r.judgments.values())
                for r in state.verification_records
                if r.judgments
            )
            if curated and has_positive:
                return "good"
            return "ambiguous"
        if action.action_type == CapabilityActionType.VERIFY_CLAIM:
            docs = action.arguments.get("doc_ids") or []
            claim = str(action.arguments.get("claim", "")).strip()
            if not docs or not claim:
                return "bad"
            if docs and all(str(d) in visible for d in docs):
                return "good"
            if docs:
                return "bad"
            return "ambiguous"
        return "ambiguous"

    if module_id == "evidence_state":
        if action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
        }:
            add_ids = action.arguments.get("add_ids") or []
            if not isinstance(add_ids, list):
                add_ids = []
            if add_ids and any(str(d) not in pool and str(d) not in curated for d in add_ids):
                return "bad"
            if add_ids:
                return "good"
            return "ambiguous"
        if action.action_type == CapabilityActionType.SEARCH:
            if (state.repeated_query_score or 0) >= 0.8:
                return "bad"
            return "ambiguous"
        if stopping and not curated:
            return "bad"
        return "ambiguous"

    return "ambiguous"


# Display names for the capability audit table.
CAPABILITY_DISPLAY = {
    "DUPLICATE_EVIDENCE": "Duplicate Evidence",
    "MISSING_DIRECT_SUPPORT": "Missing Support",
    "MISSING_DIRECT_EVIDENCE": "Missing Support",
    "CLAIM_WITHOUT_SUPPORT": "Missing Support",
    "CONFLICTING_EVIDENCE": "Conflict",
    "UNRESOLVED_CONFLICT": "Conflict",
    "PREMATURE_STOP": "Premature Stop",
    "WRONG_CLAIM_BINDING": "Wrong Evidence Binding",
    "MISSING_CLAIM_LINK": "Wrong Evidence Binding",
    "IRRELEVANT_EVIDENCE": "Irrelevant Evidence",
    "WEAK_SUPPORT": "Weak Support",
    "INVALID_CITATION": "Invalid Citation",
    "SOURCE_NOT_VISIBLE": "Source Not Visible",
}


def summarize_capability_table(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Calls / Correct / Precision / Recall per capability display name."""
    # Aggregate by display name
    gt_calls: dict[str, int] = {}
    pred_correct: dict[str, int] = {}
    tp: dict[str, int] = {}

    for ev in events:
        local_caps = list(ev.get("local_capabilities") or [])
        reason = str(ev.get("reason_code") or "")
        mode = str(ev.get("mode") or "")
        pred_name = CAPABILITY_DISPLAY.get(reason)
        local_names = {
            CAPABILITY_DISPLAY[c] for c in local_caps if c in CAPABILITY_DISPLAY
        }

        for name in local_names:
            gt_calls[name] = gt_calls.get(name, 0) + 1

        if mode == "correct" and pred_name:
            pred_correct[pred_name] = pred_correct.get(pred_name, 0) + 1
            if pred_name in local_names:
                tp[pred_name] = tp.get(pred_name, 0) + 1

    names = sorted(set(gt_calls) | set(pred_correct))
    rows: list[dict[str, Any]] = []
    for name in names:
        calls = int(gt_calls.get(name, 0))
        correct = int(pred_correct.get(name, 0))
        hits = int(tp.get(name, 0))
        precision = hits / correct if correct else 0.0
        recall = hits / calls if calls else 0.0
        rows.append(
            {
                "capability": name,
                "calls": calls,
                "correct": correct,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
            }
        )
    # Prefer a stable report order matching the target table
    preferred = [
        "Duplicate Evidence",
        "Missing Support",
        "Conflict",
        "Premature Stop",
        "Wrong Evidence Binding",
        "Irrelevant Evidence",
        "Weak Support",
        "Invalid Citation",
        "Source Not Visible",
    ]
    order = {n: i for i, n in enumerate(preferred)}
    rows.sort(key=lambda r: (order.get(r["capability"], 99), r["capability"]))
    return rows


@dataclass
class ModuleAudit:
    calls: int = 0
    endorse: int = 0
    correct: int = 0
    noop: int = 0
    masked: int = 0
    # local labels
    n_good: int = 0
    n_bad: int = 0
    n_ambiguous: int = 0
    # intervention
    correct_and_bad: int = 0  # TP-ish for precision numerator
    correct_and_good: int = 0
    bad_and_correct: int = 0  # recall numerator (= correct_and_bad)
    # conditioned
    correct_given_bad: int = 0
    correct_given_good: int = 0

    def as_dict(self) -> dict[str, Any]:
        n = max(self.calls, 1)
        n_bad = max(self.n_bad, 1)
        n_good = max(self.n_good, 1)
        n_correct = max(self.correct, 1)
        return {
            "calls": self.calls,
            "endorse_rate": self.endorse / n,
            "correct_rate": self.correct / n,
            "noop_rate": self.noop / n,
            "masked_rate": self.masked / n,
            "n_good": self.n_good,
            "n_bad": self.n_bad,
            "n_ambiguous": self.n_ambiguous,
            "P_CORRECT_given_bad": self.correct_given_bad / n_bad,
            "P_CORRECT_given_good": self.correct_given_good / n_good,
            "intervention_precision": self.correct_and_bad / n_correct,
            "intervention_recall": self.bad_and_correct / n_bad,
        }


@dataclass
class GlobalStats:
    num_states: int = 0
    num_endorse: int = 0
    num_correct: int = 0
    num_noop: int = 0
    num_masked: int = 0
    by_module: dict[str, ModuleAudit] = field(
        default_factory=lambda: defaultdict(ModuleAudit)
    )
    episodes: int = 0

    def record(
        self,
        *,
        module_id: str,
        mode: GuidanceMode,
        masked: bool,
        local: LocalLabel,
    ) -> None:
        self.num_states += 1
        mc = self.by_module[module_id]
        mc.calls += 1
        if local == "good":
            mc.n_good += 1
        elif local == "bad":
            mc.n_bad += 1
        else:
            mc.n_ambiguous += 1

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
            if local == "bad":
                mc.correct_and_bad += 1
                mc.bad_and_correct += 1
                mc.correct_given_bad += 1
            elif local == "good":
                mc.correct_and_good += 1
                mc.correct_given_good += 1
        else:
            self.num_noop += 1
            mc.noop += 1

    def to_dict(self) -> dict[str, Any]:
        n = max(self.num_states, 1)
        return {
            "num_states": self.num_states,
            "num_endorse": self.num_endorse,
            "num_correct": self.num_correct,
            "num_noop": self.num_noop,
            "num_masked": self.num_masked,
            "endorse_rate": self.num_endorse / n,
            "correct_rate": self.num_correct / n,
            "noop_rate": self.num_noop / n,
            "episodes": self.episodes,
            "Evidence": self.by_module["evidence_state"].as_dict(),
            "Verification": self.by_module["verification"].as_dict(),
        }


def _selector(scope: dict[str, Any]) -> RuleBasedCriticalStateSelector:
    sel = scope.get("selector") or {}
    mods = scope.get("modules") or {}
    return RuleBasedCriticalStateSelector(
        SelectorConfig(
            before_stop=bool(sel.get("before_stop", True)),
            after_curate=bool(sel.get("after_curate", True)),
            after_verify=bool(sel.get("after_verify", True)),
            after_review=bool(sel.get("after_review", True)),
            after_pool_growth=bool(sel.get("after_pool_growth", True)),
            evidence_enabled=bool(mods.get("evidence_state", True)),
            verification_enabled=bool(mods.get("verification", True)),
            budget_enabled=False,
        )
    )


def audit_turn(
    turn: ChatTurnRecord,
    *,
    registry,
    selector: RuleBasedCriticalStateSelector,
    router: GuidanceRouter,
    stats: GlobalStats,
    events: list[dict[str, Any]],
) -> None:
    state = turn.decision_state
    action = turn.student_action
    module_ids = selector.select(state, action)
    for mid in module_ids:
        if not registry.has(mid):
            continue
        module = registry.get(mid)
        artifact = module.analyze(state, action)
        if artifact.mode == GuidanceMode.CORRECT:
            artifact = fill_recommended_action(state, artifact)
        artifact, vis = mask_artifact_if_invalid(state, artifact)
        masked = (not vis.valid) or bool(artifact.metadata.get("masked"))
        decision = router.route(state, artifact, module=module)
        local = label_local_decision(state, action, mid)
        local_caps = label_local_capabilities(state, action, mid)
        stats.record(
            module_id=mid,
            mode=decision.mode,
            masked=masked,
            local=local,
        )
        events.append(
            {
                "schema_version": "scope.audit_event.v3",
                "query_id": state.task_id,
                "episode_id": state.episode_id,
                "event_id": state.event_id or f"{state.episode_id}:{state.turn_id}:{mid}",
                "turn_id": state.turn_id,
                "module_id": mid,
                "capability_id": decision.artifact.capability_id
                or decision.artifact.resolved_capability().value,
                "mode": decision.mode.value,
                "reason_code": decision.artifact.reason_code,
                "masked": masked,
                "local_label": local,
                "local_capabilities": local_caps,
                "student_action": action.action_type.value,
                "action_arguments": dict(action.arguments),
                "add_ids": list(action.arguments.get("add_ids") or [])
                if isinstance(action.arguments.get("add_ids"), list)
                else [],
                "recommended_action": (
                    decision.artifact.recommended_action.to_dict()
                    if decision.artifact.recommended_action is not None
                    else None
                ),
                "pool": len(state.pool_document_ids),
                "curated": len(state.curated_document_ids),
                "n_verify": len(state.verification_records),
                "n_claims": len(state.evidence_claims),
                "query": state.query[:500],
                "rendered_context": (state.rendered_context or "")[:4000],
                "verification_records": [
                    {
                        "turn_id": r.turn_id,
                        "claim": r.claim[:200],
                        "document_ids": list(r.document_ids),
                        "judgments": dict(r.judgments),
                    }
                    for r in state.verification_records[-5:]
                ],
                # V3 training payload (full enough to rebuild DecisionSupervisionSampleV3)
                "decision_state": state.to_dict(),
                "artifact": decision.artifact.to_dict(),
                "student_action_struct": action.to_dict(),
            }
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCOPE chat-online DecisionState audit")
    p.add_argument("--config", default="configs/scope/shadow_audit_m1_m2.yaml")
    p.add_argument(
        "--harness-config",
        default=str(config_path("modules_full.yaml")),
    )
    p.add_argument("--output-dir", default="outputs/scope_chat_decision_audit")
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--limit", type=int, default=40, help="Sanity size (20–50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--vllm-port", type=int, default=8773)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--vllm-model-name", default="scope-chat-audit")
    p.add_argument(
        "--manage-vllm",
        action="store_true",
        default=False,
        help="Start vLLM from Python (no tool-call flags; prefer shell launcher)",
    )
    p.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--retrieval", default="bm25", choices=["bm25", "chroma"])
    p.add_argument("--bm25-index-path", default=None)
    p.add_argument("--reranker", default="none")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--split", default="all")
    p.add_argument(
        "--query-timeout-s",
        type=float,
        default=600.0,
        help="Per-query wall timeout (seconds); skip on hang",
    )
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scope = scope_section(load_scope_config(args.config))
    harness_cfg = load_harness_config(args.harness_config)
    apply_harness_config(harness_cfg)
    harness_cfg.save_resolved(out_dir / "harness_resolved_config.yaml")

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
    print(f"[chat-audit] n_queries={len(records)} out={out_dir}", flush=True)

    episode_jsonl = out_dir / "chat_audit_episodes.jsonl"
    events_jsonl = out_dir / "chat_decision_audit_events.jsonl"
    done = load_completed_query_ids(episode_jsonl) if args.resume else set()
    pending = [r for r in records if r.query_id not in done]
    if not args.resume:
        # Fresh run: truncate incremental event log
        events_jsonl.write_text("", encoding="utf-8")
        if episode_jsonl.exists():
            episode_jsonl.write_text("", encoding="utf-8")
        done = set()
        pending = list(records)

    registry = build_default_registry(
        evidence_state=True, verification=True, budget_control=False
    )
    selector = _selector(scope)
    router = GuidanceRouter()
    stats = GlobalStats()
    events: list[dict[str, Any]] = []

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"

    # Point llm_env at local vLLM (process-local env vars; clear cache).
    os.environ["base_url"] = base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = args.vllm_model_name
    from harness.llm_env import get_llm_settings

    get_llm_settings.cache_clear()

    try:
        if args.manage_vllm and args.vllm_url is None:
            print(
                f"[chat-audit] Starting vLLM TP={args.tensor_parallel_size} "
                f"with tool-call-parser=hermes at {base_url}",
                flush=True,
            )
            # start_vllm_server doesn't pass tool flags — start via subprocess wrapper in shell.
            # Here we assume shell launcher starts vLLM; if manage_vllm, use extended start.
            from training.opd.vllm_server import start_vllm_server as _start

            # Monkey-patch: extend command via env already set in launcher.
            # Fallback: start without tool flags only if launcher didn't.
            import urllib.request

            ready = False
            try:
                with urllib.request.urlopen(f"{base_url}/models", timeout=3) as resp:
                    ready = resp.status == 200
            except Exception:
                ready = False
            if not ready:
                # Minimal start; prefer launcher that adds --enable-auto-tool-choice
                vllm_handle = _start(
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
            print(f"[chat-audit] vLLM ready: {base_url}", flush=True)

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
        stats_lock = asyncio.Lock()
        completed = len(records) - len(pending)

        async def _one(record) -> None:
            nonlocal completed
            ep_events: list[dict[str, Any]] = []
            ep_stats = GlobalStats()
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
                        audit_turn(
                            turn,
                            registry=registry,
                            selector=selector,
                            router=router,
                            stats=ep_stats,
                            events=ep_events,
                        )

                    result = await asyncio.wait_for(
                        driver.run(on_critical_turn=_on_turn),
                        timeout=float(args.query_timeout_s),
                    )
                    # If the policy never issued stop/answer/verify, still audit the
                    # terminal "准备停止" state (max-turns / silent end). This is the
                    # key Premature-Stop / Missing-Support slice.
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
                        # Always audit a terminal stop — including after verify —
                        # so positive verification_records can yield ENDORSE.
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
                        audit_turn(
                            terminal_turn,
                            registry=registry,
                            selector=selector,
                            router=router,
                            stats=ep_stats,
                            events=ep_events,
                        )
                    ep_stats.episodes = 1
            except asyncio.TimeoutError:
                print(
                    f"[chat-audit] TIMEOUT query={record.query_id} "
                    f"after {args.query_timeout_s}s",
                    flush=True,
                )
                result = {
                    "query_id": record.query_id,
                    "turns": 0,
                    "recall": 0.0,
                    "final_answer_recall": 0.0,
                    "n_curated": 0,
                    "n_pool": 0,
                    "error": True,
                    "error_msg": f"query_timeout_{args.query_timeout_s}s",
                }
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[chat-audit] ERROR query={record.query_id}: {exc}",
                    flush=True,
                )
                result = {
                    "query_id": record.query_id,
                    "turns": 0,
                    "recall": 0.0,
                    "final_answer_recall": 0.0,
                    "n_curated": 0,
                    "n_pool": 0,
                    "error": True,
                    "error_msg": str(exc)[:500],
                }

            async with stats_lock:
                stats.num_states += ep_stats.num_states
                stats.num_endorse += ep_stats.num_endorse
                stats.num_correct += ep_stats.num_correct
                stats.num_noop += ep_stats.num_noop
                stats.num_masked += ep_stats.num_masked
                stats.episodes += 1
                for mid, mc in ep_stats.by_module.items():
                    d = stats.by_module[mid]
                    for attr in (
                        "calls",
                        "endorse",
                        "correct",
                        "noop",
                        "masked",
                        "n_good",
                        "n_bad",
                        "n_ambiguous",
                        "correct_and_bad",
                        "correct_and_good",
                        "bad_and_correct",
                        "correct_given_bad",
                        "correct_given_good",
                    ):
                        setattr(d, attr, getattr(d, attr) + getattr(mc, attr))
                events.extend(ep_events)

            row = {
                "query_id": record.query_id,
                "turns": result.get("turns", 0),
                "recall": result.get("recall", 0.0),
                "final_answer_recall": result.get("final_answer_recall", 0.0),
                "n_curated": result.get("n_curated", 0),
                "n_pool": result.get("n_pool", 0),
                "error": result.get("error", False),
                "error_msg": result.get("error_msg"),
                "n_shadow_events": len(ep_events),
            }
            async with write_lock:
                completed += 1
                with episode_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                # Persist shadow events incrementally (survive crashes / hangs)
                if ep_events:
                    with events_jsonl.open("a", encoding="utf-8") as fh:
                        for ev in ep_events:
                            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                if args.log_every and completed % args.log_every == 0:
                    s = stats.to_dict()
                    print(
                        f"[chat-audit] {completed}/{len(records)} "
                        f"last={record.query_id} turns={row['turns']} "
                        f"correct_rate={s['correct_rate']:.2f} "
                        f"V.correct={s['Verification']['correct_rate']:.2f} "
                        f"E.correct={s['Evidence']['correct_rate']:.2f}",
                        flush=True,
                    )

        await asyncio.gather(*[_one(r) for r in pending])

        # Rebuild capability table from on-disk events (includes prior resume chunks)
        all_events: list[dict[str, Any]] = []
        if events_jsonl.exists():
            with events_jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not all_events:
            all_events = events

        summary = stats.to_dict()
        summary["mode"] = "chat_online_decision_audit"
        summary["n_queries"] = len(records)
        summary["n_events"] = len(all_events)
        summary["model_path"] = args.model_path
        capability_table = summarize_capability_table(all_events)
        summary["capability_table"] = capability_table
        (out_dir / "chat_decision_audit_stats.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Also write a compact markdown-friendly table
        table_path = out_dir / "capability_audit_table.md"
        lines = [
            "| Capability | Calls | Correct | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in capability_table:
            lines.append(
                f"| {row['capability']} | {row['calls']} | {row['correct']} | "
                f"{row['precision']:.2f} | {row['recall']:.2f} |"
            )
        table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        def _pct_block(name: str, d: dict[str, Any]) -> str:
            if d.get("calls", 0) == 0:
                return f"{name}\n--------------------------------\n(no calls)\n"
            return (
                f"{name}\n"
                f"--------------------------------\n"
                f"ENDORSE     {100 * d['endorse_rate']:.0f}%\n"
                f"CORRECT     {100 * d['correct_rate']:.0f}%\n"
                f"NOOP        {100 * d['noop_rate']:.0f}%\n"
                f"MASK         {100 * d['masked_rate']:.0f}%\n"
                f"P(CORRECT|bad)={d['P_CORRECT_given_bad']:.3f}  "
                f"P(CORRECT|good)={d['P_CORRECT_given_good']:.3f}\n"
                f"Intervention Precision={d['intervention_precision']:.3f}  "
                f"Recall={d['intervention_recall']:.3f}\n"
            )

        pretty = (
            "\n[chat-audit] DONE — DecisionState online (chat, no Harmony)\n\n"
            + _pct_block("Verification", summary["Verification"])
            + "\n"
            + _pct_block("Evidence", summary["Evidence"])
            + "\nCapability table\n--------------------------------\n"
            + "\n".join(lines)
            + "\n"
        )
        print(pretty, flush=True)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if vllm_handle is not None:
            print("[chat-audit] Stopping vLLM ...", flush=True)
            vllm_handle.stop()


def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
