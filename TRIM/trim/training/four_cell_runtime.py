"""Formal Before / RL / PURE / RL+OPD loop for sr_opd_ce + CISPO.

Default backend is vLLM batched rollout + HF train (Scheme A). Each hybrid
substep is CISPO FB → CE FB → one optim_step. RL / RL+OPD re-rollout after
every optimizer step. Teacher is a side branch and must not change RL rewards.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Callable

from trim.adapters.components import (
    all_component_ids,
    coalition_minus_mask,
    default_component_ids,
    full_mask,
    minus_mask,
    zero_mask,
)
from trim.upstream_harness1.v8d_flags import all_enabled_mask
from trim.adapters.harness_profiles import infer_harness_from_ids, is_harness_g
from trim.eval.adapter_reload_audit import audit_saved_adapter, write_reload_audit
from trim.eval.browsecomp_retrieval import RetrievalBackend, hits_to_doc_store, open_retrieval
from trim.eval.official_query_pool import (
    BCPLUS_TEST,
    BCPLUS_TOTAL,
    BCPLUS_TRAIN,
    SCORE_SPLIT_166,
    SCORE_SPLIT_830,
    attach_bcp_fields,
    is_full_score_split,
    load_bcplus_830_full,
    load_bcplus_830_split,
    load_train_queries,
    official_test_subset,
    overlap_ids,
)
from trim.eval.sec_corpus import (
    SEC_TRAIN_POOL_NAME,
    attach_sec_doc_stores,
    corpus_bm25_index,
    default_sec_corpus_root,
    default_sec_rl_data,
    load_sec_rl_queries,
    open_sec_retrieval,
)
from trim.eval.sr_opd_four_cell_eval import (
    pack_closed_loop_summary,
    search_metrics,
    split_summaries,
    summarize_traces,
    write_eval_outputs,
)
from trim.training.frozen_state_loader import (
    doc_store_from_points,
    groups_from_frozen_points,
    load_train_states,
)
from trim.state.snapshot import capture_snapshot
from trim.training.action_codec import (
    HARNESS_G_STUDENT_NATIVE_TOOLS,
    STUDENT_NATIVE_TOOLS,
    render_action,
)
from trim.training.hf_rl_opd_client import (
    HFDebugTrainingClient,
    episode_relative_advantages,
    sample_groups_for_step,
)
from trim.training.policy_digest import adapter_digest, policy_digest_record
from trim.training.train_checkpoint import (
    append_metrics_jsonl,
    load_optimizer_bundle,
    publish_step_checkpoint,
    save_optimizer_bundle,
    save_rng_state,
)
from trim.training.train_query_sampler import QuerySampler
from trim.training.hf_rl_batch import (
    HF_DEFAULT_GROUPS_PER_STEP,
    HF_DEFAULT_HEARTBEAT_EVERY,
    HF_DEFAULT_MICRO_BATCH,
    log_train,
)
from trim.training.on_policy_collector import filter_component_states, write_collected_states
from trim.training.opd_dataset import render_student_prompt
from trim.training.rl_opd_types import (
    PROTOCOL_COMPLETE_RL_OPD,
    TRAINING_MODE_PURE_OPD,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
    TRAINING_MODE_SCAPE_RL,
    TRAINING_MODE_SCAPE_SEED,
    HybridRolloutGroup,
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP,
    SCAPE_RL_LAMBDA_OPD,
    SCAPE_RL_OPD_GATE_BETA,
    StudentDecisionPoint,
    uses_sampled_opd,
)
from trim.training.parse_rollout_action import parse_generated_action
from trim.training.teacher_isolation import COMPONENT_KIND, run_teacher_branch_isolated
from trim.training.auto_populate_teacher import teacher_events_from_point as auto_populate_events_from_point
from trim.training.sentence_compress_teacher import teacher_events_from_point
from trim.training.token_budget_marker_teacher import teacher_events_from_point as token_budget_marker_events_from_point
from trim.training.adaptive_rerank_teacher import teacher_events_from_point as adaptive_rerank_events_from_point
from trim.training.verify_tool_teacher import teacher_events_from_point as verify_tool_events_from_point
from trim.training.harness_g_teacher import teacher_events_from_point_for as harness_g_events_from_point
from trim.training.tinker_rl_opd_trainer import hybrid_train_substep, prepare_hybrid_batch

def stable_rollout_seed(base: int, tag: str) -> int:
    digest = hashlib.sha256(f"{int(base)}:{tag}".encode("utf-8")).hexdigest()
    return int(base) + (int(digest[:8], 16) % 100_000)


CELLS = ("teacher", "before", "pure_opd", "rl_opd")
TeacherFn = Callable[[StudentDecisionPoint], list[Any]]

TEACHER_REGISTRY: dict[str, TeacherFn] = {
    "auto_populate_first_search": auto_populate_events_from_point,
    "sentence_compress": teacher_events_from_point,
    "token_budget_marker": token_budget_marker_events_from_point,
    "adaptive_rerank_instruction": adaptive_rerank_events_from_point,
    "verify_tool": verify_tool_events_from_point,
    "answer_with": lambda point: harness_g_events_from_point("answer_with", point),
    "bridge_entities": lambda point: harness_g_events_from_point("bridge_entities", point),
    "entity_synonyms": lambda point: harness_g_events_from_point("entity_synonyms", point),
    "sentence_neighbors": lambda point: harness_g_events_from_point("sentence_neighbors", point),
    "hybrid_init_retrieve": lambda point: harness_g_events_from_point("hybrid_init_retrieve", point),
    "snc_frontier": lambda point: harness_g_events_from_point("snc_frontier", point),
    "invalid_target_filter": lambda point: harness_g_events_from_point("invalid_target_filter", point),
    "lookup_dedup": lambda point: harness_g_events_from_point("lookup_dedup", point),
}


def component_ids_of(value: Any, *, harness: str | None = None) -> list[str]:
    """Parse a single id, comma-separated coalition, or `zero` / `all` / `default`."""
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
    else:
        text = str(value or "").replace(";", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
    lowered = [p.lower() for p in parts]
    if any(p == "zero" for p in lowered):
        if len(parts) != 1 or parts[0].lower() != "zero":
            raise SystemExit("component zero cannot be mixed with other ids")
        return []
    if any(p == "default" for p in lowered):
        if len(parts) != 1 or parts[0].lower() != "default":
            raise SystemExit("component default cannot be mixed with other ids")
        resolved = harness or infer_harness_from_ids(parts)
        return default_component_ids(resolved)
    if any(p == "all" for p in lowered):
        if len(parts) != 1 or parts[0].lower() != "all":
            raise SystemExit("component all cannot be mixed with other ids")
        resolved = harness or infer_harness_from_ids(parts)
        return list(all_component_ids(resolved))
    resolved = harness or infer_harness_from_ids(parts)
    known = set(all_component_ids(resolved))
    unknown = [p for p in parts if p not in known]
    if unknown:
        raise SystemExit(
            f"unknown component id(s) {unknown}; allowed: zero, all, default, or {list(all_component_ids(resolved))}"
        )
    if not parts:
        raise SystemExit("component id is empty; pass zero to disable all advanced components")
    return parts


def student_mask_for(component_id: Any, *, harness: str | None = None) -> dict[str, bool]:
    resolved = harness or infer_harness_from_ids(component_id)
    ids = component_ids_of(component_id, harness=resolved)
    if not ids:
        return zero_mask(resolved)
    if len(ids) == 1:
        return minus_mask(ids[0], harness=resolved)
    return coalition_minus_mask(ids, harness=resolved)


def teacher_mask_for(component_id: Any, *, harness: str | None = None) -> dict[str, bool]:
    resolved = harness or infer_harness_from_ids(component_id)
    ids = component_ids_of(component_id, harness=resolved)
    if not ids:
        return zero_mask(resolved)
    if ids == list(all_component_ids(resolved)):
        return all_enabled_mask(resolved)
    mask = full_mask(resolved)
    for cid in ids:
        mask[cid] = True
    return mask


def resolved_rollout_mask(
    component_id: Any,
    *,
    harness: str | None = None,
    harness_mask: dict[str, bool] | None = None,
    teacher_mode: bool = False,
) -> dict[str, bool]:
    """Mask written into live env state for a rollout.

    An explicit ``harness_mask`` wins. Otherwise teacher cells use H_full and
    student / RL / TRIM rollouts use H_min. Never pass ``None`` through to
    ``new_state``: that silently becomes zero and desyncs from snapshots.
    """
    if harness_mask is not None:
        return dict(harness_mask)
    if teacher_mode:
        return teacher_mask_for(component_id, harness=harness)
    return student_mask_for(component_id, harness=harness)


def teacher_action_from_point(
    point: StudentDecisionPoint,
    component_id: str,
    *,
    harness: str | None = None,
    teacher_kind: str = "upstream",
) -> dict[str, Any]:
    """First student-legal teacher action for a heuristic teacher-cell turn."""
    fn = teacher_for(component_id, harness=harness, teacher_kind=teacher_kind)

    def _run(forked_snap: Any) -> dict[str, Any]:
        forked_point = replace(point, pre_action_snapshot=forked_snap)
        events = fn(forked_point) if fn is not None else []
        action_event = next((e for e in events if getattr(e, "action_name", None)), None)
        if action_event is None:
            q = str((forked_snap.working_memory or {}).get("query") or "")
            return {"name": "search_corpus", "arguments": {"query": q}}
        return {
            "name": str(action_event.action_name),
            "arguments": dict(action_event.arguments or {}),
        }

    action, _original = run_teacher_branch_isolated(point.pre_action_snapshot, _run)
    return action


def generic_teacher_events_from_wm(
    wm: dict[str, Any],
    component_id: str,
    *,
    turn_id: int = 0,
) -> list[Any]:
    """Fallback Teacher for taxonomy ids that do not have a dedicated side-branch."""
    from trim.training.opd_events import model_action, obs_transform
    from trim.training.sentence_compress_teacher import documents_from_wm, score_doc

    q = str(wm.get("query") or "")
    docs = documents_from_wm(wm)
    events = [
        obs_transform(
            component_id,
            turn_id=turn_id,
            observation={"owner": "teacher_full", "generic_teacher": True},
            visible_to_student=False,
            metadata={"owner": "teacher_full", "student_must_not_see": True},
        )
    ]
    curated = {str(x) for x in (wm.get("curated_ids") or [])}
    if not docs:
        events.append(
            model_action(
                "search_corpus",
                {"query": q},
                turn_id=turn_id,
                component_id=component_id,
            )
        )
        return events
    ranked = sorted(
        ((did, text) for did, text in docs if did not in curated),
        key=lambda it: (-score_doc(q, it[1]), it[0]),
    )
    add_ids = [did for did, _ in ranked[:2]] or [docs[0][0]]
    events.append(
        model_action(
            "curate",
            {"add_ids": add_ids, "remove_ids": []},
            turn_id=turn_id,
            component_id=component_id,
        )
    )
    return events


def _teacher_fn_for_one(component_id: str, *, teacher_kind: str = "upstream") -> TeacherFn:
    registered = TEACHER_REGISTRY.get(component_id)
    if registered is not None:
        return registered
    if teacher_kind == "heuristic_generic":
        def _generic(point: StudentDecisionPoint) -> list[Any]:
            wm = point.pre_action_snapshot.working_memory
            events = generic_teacher_events_from_wm(wm, component_id, turn_id=int(point.turn_id))
            for event in events:
                meta = getattr(event, "metadata", None)
                if isinstance(meta, dict):
                    meta["teacher_kind"] = "heuristic_generic"
            return events

        return _generic

    def _skip(point: StudentDecisionPoint) -> list[Any]:
        from trim.training.opd_events import obs_transform

        kind = COMPONENT_KIND.get(component_id, "unknown")
        return [
            obs_transform(
                component_id,
                turn_id=int(point.turn_id),
                observation={
                    "owner": "teacher_full",
                    "skip_reason": "no_upstream_side_branch",
                    "component_kind": kind,
                },
                visible_to_student=False,
                metadata={
                    "teacher_kind": "skip_unregistered",
                    "component_kind": kind,
                    "not_a_continuous_teacher_rollout": True,
                },
            )
        ]

    return _skip


def teacher_for(component_id: str, *, harness: str | None = None, teacher_kind: str = "upstream") -> TeacherFn | None:
    ids = component_ids_of(component_id, harness=harness)
    if not ids:
        return _teacher_fn_for_one("zero", teacher_kind=teacher_kind)
    fns = [_teacher_fn_for_one(cid, teacher_kind=teacher_kind) for cid in ids]
    if len(fns) == 1:
        return fns[0]

    def _combined(point: StudentDecisionPoint) -> list[Any]:
        events: list[Any] = []
        for fn in fns:
            events.extend(fn(point))
        for event in events:
            meta = getattr(event, "metadata", None)
            if isinstance(meta, dict):
                meta["combined_independent_events"] = True
                meta["not_a_continuous_teacher_rollout"] = True
        return events

    return _combined


def teacher_events_from_wm_for(component_id: str, wm: dict[str, Any]) -> list[Any]:
    from trim.training.adaptive_rerank_teacher import teacher_events_from_wm as adaptive_rerank_events_from_wm
    from trim.training.auto_populate_teacher import teacher_events_from_wm as auto_populate_events_from_wm
    from trim.training.sentence_compress_teacher import teacher_events_from_wm
    from trim.training.token_budget_marker_teacher import teacher_events_from_wm as token_budget_marker_events_from_wm
    from trim.training.harness_g_teacher import teacher_events_from_wm as harness_g_events_from_wm
    from trim.training.verify_tool_teacher import teacher_events_from_wm as verify_tool_events_from_wm

    builders = {
        "auto_populate_first_search": auto_populate_events_from_wm,
        "sentence_compress": teacher_events_from_wm,
        "token_budget_marker": token_budget_marker_events_from_wm,
        "adaptive_rerank_instruction": adaptive_rerank_events_from_wm,
        "verify_tool": verify_tool_events_from_wm,
        "answer_with": lambda wm: harness_g_events_from_wm("answer_with", wm),
        "bridge_entities": lambda wm: harness_g_events_from_wm("bridge_entities", wm),
        "entity_synonyms": lambda wm: harness_g_events_from_wm("entity_synonyms", wm),
        "sentence_neighbors": lambda wm: harness_g_events_from_wm("sentence_neighbors", wm),
        "hybrid_init_retrieve": lambda wm: harness_g_events_from_wm("hybrid_init_retrieve", wm),
        "snc_frontier": lambda wm: harness_g_events_from_wm("snc_frontier", wm),
        "invalid_target_filter": lambda wm: harness_g_events_from_wm("invalid_target_filter", wm),
        "lookup_dedup": lambda wm: harness_g_events_from_wm("lookup_dedup", wm),
    }
    events: list[Any] = []
    ids = component_ids_of(component_id, harness=infer_harness_from_ids(component_id))
    if not ids:
        return generic_teacher_events_from_wm(wm, "zero")
    for cid in ids:
        fn = builders.get(cid)
        if fn is None:
            events.extend(generic_teacher_events_from_wm(wm, cid))
        else:
            events.extend(fn(wm))
    return events


def cell_lambda(name: str, lambda_opd: float) -> float:
    if name in {"before", "rl"}:
        return 0.0
    return float(lambda_opd)


def cells_for_mode(training_mode: str | None, *, train_only: bool = False) -> tuple[str, ...]:
    """Four-cell protocol, or the single training cell used by ``run_train.py``."""
    only = {
        TRAINING_MODE_RL: ("rl",),
        TRAINING_MODE_PURE_OPD: ("pure_opd",),
        TRAINING_MODE_RL_OPD: ("rl_opd",),
        TRAINING_MODE_SCAPE_RL: ("scape_rl",),
        TRAINING_MODE_SCAPE_SEED: ("scape_seed",),
        "rl_only": ("rl",),
        "pure_opd_only": ("pure_opd",),
        "rl_opd_only": ("rl_opd",),
        "scape_rl_only": ("scape_rl",),
        "scape_seed_only": ("scape_seed",),
    }
    if train_only and training_mode in only:
        return only[training_mode]
    if training_mode in {None, "", "four_cell"}:
        return CELLS
    if training_mode == TRAINING_MODE_RL:
        return ("before", "rl")
    if training_mode == TRAINING_MODE_PURE_OPD:
        return ("before", "pure_opd")
    if training_mode == TRAINING_MODE_RL_OPD:
        return ("before", "rl_opd")
    if training_mode == TRAINING_MODE_SCAPE_RL:
        return ("before", "scape_rl")
    if training_mode == TRAINING_MODE_SCAPE_SEED:
        return ("before", "scape_seed")
    if training_mode in only:
        return only[training_mode]
    return CELLS


def is_scape_rl_mode(args: argparse.Namespace | None = None, *, training_mode: str | None = None) -> bool:
    mode = training_mode if training_mode is not None else str(getattr(args, "training_mode", "") or "")
    return mode == TRAINING_MODE_SCAPE_RL


def uses_sec_train_data(args: argparse.Namespace | None = None, *, training_mode: str | None = None) -> bool:
    """True when the train query pool is Harness-1 SEC RL (~3453).

    CLI ``--train-data sec`` (the default) selects this pool. If ``train_data``
    is unset, keep the legacy rule: only scape+rl used SEC queries.
    """
    raw = getattr(args, "train_data", None) if args is not None else None
    if raw not in {None, ""}:
        key = str(raw).strip().lower().replace("-", "_").replace(" ", "")
        if key in {"bcplus_train_664", "bcplus_664", "664", "bcplus", "bcplus_train"}:
            return False
        if key in {"sec", "sec_rl", "harness_1_rl_data", "harness1_rl_data", "harness_1_rl", "rl_data"}:
            return True
        raise ValueError(f"unknown train_data={raw!r}; use sec or bcplus_train_664")
    return is_scape_rl_mode(args, training_mode=training_mode)


def is_seed_scale_mode(args: argparse.Namespace | None = None, *, training_mode: str | None = None) -> bool:
    mode = training_mode if training_mode is not None else str(getattr(args, "training_mode", "") or "")
    return mode in {TRAINING_MODE_SCAPE_RL, TRAINING_MODE_SCAPE_SEED}


def uses_bcplus_830_eval(args: argparse.Namespace) -> bool:
    split = str(getattr(args, "score_split", "") or "")
    if is_full_score_split(split):
        return True
    if split == SCORE_SPLIT_166:
        return False
    return uses_sec_train_data(args)


def _git_provenance() -> dict[str, Any]:
    import subprocess

    root = Path(__file__).resolve().parents[2]
    out: dict[str, Any] = {"repo_root": str(root)}
    try:
        out["sha"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        out["dirty"] = bool(dirty.strip())
        if dirty.strip():
            out["dirty_diff_lines"] = len(dirty.strip().splitlines())
    except Exception:
        out["sha"] = None
        out["dirty"] = None
    return out


def _resolved_vllm_config(args: argparse.Namespace, *, tp: int) -> dict[str, Any]:
    return {
        "rollout_backend": str(getattr(args, "rollout_backend", "vllm") or "vllm"),
        "tensor_parallel_size": int(tp),
        "max_model_len": int(getattr(args, "max_model_len", 8192) or 8192),
        "max_new_tokens": int(getattr(args, "max_new_tokens", 2048) or 2048),
        "max_num_seqs": int(getattr(args, "max_num_seqs", 256) or 256),
        "gpu_memory_utilization": float(getattr(args, "gpu_memory_utilization", 0.90) or 0.90),
        "enforce_eager": bool(getattr(args, "enforce_eager", True)),
        "generate_timeout_s": float(getattr(args, "vllm_generate_timeout_s", 3600.0) or 3600.0),
        "disable_custom_all_reduce": getattr(args, "vllm_disable_custom_all_reduce", None),
    }


def build_manifest(args: argparse.Namespace, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = getattr(args, "training_mode", "four_cell")
    lam = 0.0 if mode == TRAINING_MODE_RL else float(args.lambda_opd)
    opd_loss = str(getattr(args, "opd_loss", None) or "sr_opd_ce")
    if mode == TRAINING_MODE_SCAPE_RL:
        opd_loss = str(getattr(args, "opd_loss", None) or OPD_LOSS_SAMPLED_GAP)
    elif mode == TRAINING_MODE_SCAPE_SEED:
        opd_loss = str(getattr(args, "opd_loss", None) or OPD_LOSS_PROJECTED_GAP)
    return {
        "training_mode": mode,
        "component": args.component,
        "harness": getattr(args, "harness", None) or infer_harness_from_ids(args.component),
        "component_ids": component_ids_of(
            args.component,
            harness=getattr(args, "harness", None),
        ),
        "target_component": args.component,
        "rl_loss_fn": "cispo",
        "opd_loss": opd_loss,
        "lambda_opd": lam,
        "opd_gate_beta": float(getattr(args, "opd_gate_beta", SCAPE_RL_OPD_GATE_BETA) or SCAPE_RL_OPD_GATE_BETA),
        "student_harness": "H_min",
        "teacher_harness": "H_full",
        "student_mask": student_mask_for(args.component, harness=getattr(args, "harness", None)),
        "teacher_mask": teacher_mask_for(args.component, harness=getattr(args, "harness", None)),
        "opd_state_source": "current_on_policy_rl_rollout",
        "joint_update_contract": "rl_fb+opd_fb+single_optim",
        "legacy_tool_token_kl_hook_used": False,
        "protocol_complete_rl_opd": mode in {"four_cell", TRAINING_MODE_RL_OPD, TRAINING_MODE_SCAPE_RL, TRAINING_MODE_SCAPE_SEED} and lam > 0,
        "protocol_name": PROTOCOL_COMPLETE_RL_OPD,
        "projection_schema_version": "scape_projection_v1",
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "train_steps": args.train_steps,
        "train_groups_per_step": int(getattr(args, "train_groups_per_step", HF_DEFAULT_GROUPS_PER_STEP) or 0),
        "train_micro_batch_size": int(getattr(args, "train_micro_batch_size", HF_DEFAULT_MICRO_BATCH) or HF_DEFAULT_MICRO_BATCH),
        "n_queries": args.n_queries,
        "opd_states_per_trajectory": args.opd_states_per_trajectory,
        "seed": args.seed,
        "base_model": args.base_model,
        "sft_adapter": getattr(args, "sft_adapter", ""),
        "scale": "smoke" if getattr(args, "smoke", False) else "full",
        "backend": (
            "vllm_rollout+hf_train"
            if str(getattr(args, "rollout_backend", "vllm") or "vllm") == "vllm"
            else "hf_debug"
        ),
        "train_backend": "hf_debug",
        "gpu_schedule": str(getattr(args, "gpu_schedule", "scheme_a") or "scheme_a"),
        "on_policy_refresh": bool(getattr(args, "on_policy_refresh", True)),
        "harmony_encoding": "o200k_harmony",
        "stop_token_ids": [200012, 200002],
        "tensor_parallel_size": getattr(args, "tensor_parallel_size", None),
        "seeds": list(getattr(args, "seeds", [args.seed])),
        "train_state_source": ("current_on_policy_rl_rollout" if args.component == "auto_populate_first_search" and not getattr(args, "train_states", None) else "train_states_5k_or_on_policy"),
        "score_split": SCORE_SPLIT_830 if uses_sec_train_data(args, training_mode=mode) else SCORE_SPLIT_166,
        "bcplus_split": (
            f"{BCPLUS_TOTAL} = {BCPLUS_TRAIN}+{BCPLUS_TEST}"
            if uses_sec_train_data(args, training_mode=mode)
            else f"{BCPLUS_TRAIN} train + {BCPLUS_TEST} test"
        ),
        "train_pool": (
            SEC_TRAIN_POOL_NAME
            if uses_sec_train_data(args, training_mode=mode)
            else "bcplus_train_664"
        ),
        "train_data": "sec" if uses_sec_train_data(args, training_mode=mode) else "bcplus_train_664",
        "rl_data": str(getattr(args, "rl_data", None) or default_sec_rl_data())
        if uses_sec_train_data(args, training_mode=mode)
        else None,
        "sec_corpus_root": str(getattr(args, "sec_corpus_root", None) or default_sec_corpus_root())
        if uses_sec_train_data(args, training_mode=mode)
        else None,
        "legacy_adapters_not_used": True,
        "train_only": bool(getattr(args, "train_only", False)),
        "max_new_tokens": int(getattr(args, "max_new_tokens", 2048) or 2048),
        "max_model_len": int(getattr(args, "max_model_len", 8192) or 8192),
        "rollout_query_batch_size": getattr(args, "rollout_query_batch_size", None),
        "git": _git_provenance(),
        **dict(extra or {}),
    }


def labeled_doc_store(row: dict[str, Any]) -> dict[str, Any]:
    gold = str((row.get("gold_docids") or ["gold"])[0])
    query = str(row.get("query") or "")
    return {
        "noise_a": {"id": "noise_a", "text": "unrelated sports scores and weather delays."},
        gold: {
            "id": gold,
            "text": (
                f"Gold evidence for the question. {query} "
                "The relevant facts appear in this document and should be curated."
            ),
        },
        "noise_b": {"id": "noise_b", "text": "background notes about exhibits and travel."},
    }


_DOC_STORE_CACHE_KEY = "_rollout_doc_store"
_DOC_STORE_CACHE_K_KEY = "_rollout_doc_store_k"


def doc_store_for_row(
    row: dict[str, Any],
    searcher: RetrievalBackend | None,
    *,
    k: int = 12,
) -> dict[str, Any]:
    """Build the per-query document store, caching the result on the row.

    Live BM25 / token-overlap search is expensive (seconds per SEC query).
    Training reuses the same ``train_rows`` across on-policy refreshes, so
    the first lookup is stored on the row and later steps skip retrieval.
    """
    k = int(k)
    cached = row.get(_DOC_STORE_CACHE_KEY)
    if isinstance(cached, dict) and int(row.get(_DOC_STORE_CACHE_K_KEY) or -1) == k:
        return dict(cached)

    def remember(store: dict[str, Any]) -> dict[str, Any]:
        stored = dict(store)
        row[_DOC_STORE_CACHE_KEY] = stored
        row[_DOC_STORE_CACHE_K_KEY] = k
        return dict(stored)

    if row.get("frozen_doc_store"):
        return remember(dict(row["frozen_doc_store"]))
    if searcher is not None and searcher.name != "none":
        hits = searcher.search(str(row.get("query") or ""), k)
        store = hits_to_doc_store(hits)
        if store:
            return remember(store)
        # Live retrieval with 0 hits is a failure — do not inject gold labels or
        # synthetic fallback docs into the episode store.
        return remember({})
    # No live searcher: start from an empty store; gold/evidence labels stay on
    # the row for scoring only (row["gold_docids"] / row["evidence_docids"]).
    return remember({})


def snap_from_state(qid: str, st: dict[str, Any], component_id: str, *, harness_mask: dict[str, bool] | None = None, teacher_mask: dict[str, bool] | None = None):
    from trim.training.upstream_train_env import is_upstream_state, sync_upstream_state

    if is_upstream_state(st):
        sync_upstream_state(st)
    curated = [str(x) for x in (st.get("curated") or {})]
    pool = [str(x) for x in (st.get("pool") or {})]
    store = st.get("doc_store") or {}
    observed_ids = list(dict.fromkeys(pool + curated))
    documents = []
    observed_store: dict[str, Any] = {}
    for did in observed_ids:
        rec = (st.get("pool") or {}).get(did) or (st.get("curated") or {}).get(did) or store.get(did)
        if rec is None:
            continue
        text = str(rec.get("text") or "") if isinstance(rec, dict) else str(rec)
        documents.append({"id": str(did), "text": text})
        observed_store[str(did)] = {
            "id": str(did),
            "text": text,
            **({k: v for k, v in rec.items() if k != "text"} if isinstance(rec, dict) else {}),
        }
    mask = resolved_rollout_mask(component_id, harness_mask=harness_mask)
    tmask = teacher_mask if teacher_mask is not None else teacher_mask_for(component_id)
    g = is_harness_g(mask=mask, component_ids=component_id)
    query_text = str(st.get("query") or "")
    wm = {
        "curated_ids": curated,
        "accessible_doc_ids": list(observed_ids),
        "pool": st.get("pool") or {},
        "documents": documents,
        "query": query_text,
        "query_text": query_text,
        "doc_store": observed_store,
        "curated_importance": dict(st.get("importance") or {}),
        "auto_populate_seed": st.get("auto_seed"),
        "evidence_graph": st.get("evidence_graph") or {},
        "token_budget_marker": st.get("token_budget_marker"),
        "rerank_instruction": st.get("rerank_instruction"),
        "runtime_effects": dict(st.get("runtime_effects") or {}),
        "step": int(st.get("step") or 0),
        "rng_state": st.get("rng_state"),
        "first_search_done": st.get("first_search_done"),
        "content_dedup_state": st.get("content_dedup_state"),
    }
    if g:
        wm.update(
            {
                "visible_sids": list(st.get("visible_sids") or []),
                "selected_sids": list(st.get("selected_sids") or []),
                "frontier_eids": list(st.get("frontier_eids") or []),
                "visited_eids": list(st.get("visited_eids") or []),
                "sentences": st.get("sentences") or {},
                "entities": st.get("entities") or {},
                "action_map": st.get("action_map") or {},
                "initialized": bool(st.get("initialized")),
            }
        )
        extra_ids = list(wm["visible_sids"]) + list(wm["selected_sids"]) + list(wm["frontier_eids"])
        wm["accessible_doc_ids"] = list(dict.fromkeys(list(wm["accessible_doc_ids"]) + extra_ids))
    return capture_snapshot(
        query_id=qid,
        query_text=query_text,
        step=int(st.get("step") or 0),
        harness_mask=mask,
        working_memory=wm,
        tool_history=list(st.get("tool_history") or []),
        observations=[],
        metadata={
            "component_id": component_id,
            "owner": "student_reduced",
            "harness": "Harness-G" if g else "Harness-1",
            "teacher_mask": dict(tmask),
            "student_mask": dict(mask),
            "display_clip_chars": None,
        },
    )


def _query_overlap(action: dict[str, Any], query: str) -> float:
    args = action.get("arguments") or {}
    blob = " ".join([str(args.get("query") or "")] + [str(x) for x in (args.get("queries") or [])]).lower()
    qset = set(re.findall(r"[a-z0-9]+", query.lower()))
    aset = set(re.findall(r"[a-z0-9]+", blob))
    if not qset or not aset:
        return 0.0
    return 0.1 * len(qset & aset) / len(qset)


def terminal_reward(st: dict[str, Any], *, query: str, gold_ids: list[str], valids: list[bool], actions: list[dict[str, Any]]) -> float:
    from trim.eval.local_search_env import curated_recall

    if not valids or not any(valids):
        return -0.2
    rec = float(curated_recall(st, gold_ids) or 0.0)
    legal = sum(1 for v in valids if v) / len(valids)
    n_unique = len({a.get("name") for a in actions if a.get("name") and a.get("name") != "unknown"})
    overlap = max((_query_overlap(a, query) for a in actions), default=0.0)
    return (
        0.15 * legal
        + 0.55 * rec
        + 0.08 * min(3, n_unique)
        + 0.08 * min(1.0, len(st.get("curated") or {}) / 2)
        + 0.06 * min(1.0, len(st.get("pool") or {}) / 3)
        + (0.08 if st.get("ended") else 0.0)
        + overlap
    )


def parse_generated_action(
    text: str,
    completion_ids: list[int] | None,
    enc,
    *,
    harness_mask: dict[str, bool] | None = None,
    teacher_mode: bool = False,
) -> tuple[dict[str, Any], bool]:
    from trim.training.parse_rollout_action import parse_generated_action as _parse

    return _parse(
        text,
        completion_ids,
        enc,
        harness_mask=harness_mask,
        teacher_mode=teacher_mode,
    )


def _maybe_empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_harmony(backend, query: str, *, enc, max_new: int, sample: bool, seed: int, prompt_ids: list[int] | None = None) -> dict[str, Any]:
    import torch
    from trim.eval.harmony_runtime import (
        build_first_turn_prompt_ids,
        decode_ids,
        stop_ids_for_tool_actions,
    )

    if prompt_ids:
        ids = list(prompt_ids)
    elif enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
        ids = enc.build_first_turn_prompt_ids(query)
    else:
        ids = build_first_turn_prompt_ids(query, enc=enc)
    stop_ids = list(getattr(enc, "stop_token_ids", None) or stop_ids_for_tool_actions(enc))
    inp = torch.tensor([ids], device=backend._device)
    attn = torch.ones_like(inp)
    cfg = getattr(backend.model, "config", None)
    old_cache = getattr(cfg, "use_cache", False) if cfg is not None else False
    if cfg is not None:
        cfg.use_cache = True
    backend.model.eval()
    try:
        kw = dict(attention_mask=attn, max_new_tokens=max_new, eos_token_id=stop_ids, pad_token_id=stop_ids[0])
        if sample:
            torch.manual_seed(seed)
            out = backend.model.generate(inp, do_sample=True, temperature=1.0, **kw)
        else:
            out = backend.model.generate(inp, do_sample=False, **kw)
    finally:
        if cfg is not None:
            cfg.use_cache = old_cache
    new_ids = out[0, inp.size(1) :].tolist()
    _maybe_empty_cache()
    return {
        "prompt_ids": ids,
        "action_ids": new_ids,
        "text": decode_ids(enc, new_ids),
        "prompt_text": decode_ids(enc, ids),
    }


def one_episode(
    backend,
    *,
    row: dict[str, Any],
    component_id: str,
    max_turns: int,
    max_new: int,
    policy_version: str,
    seed: int,
    sample: bool,
    enc,
    rollout_idx: int,
    searcher: RetrievalBackend | None = None,
    teacher_mode: bool = False,
    harness_mask: dict[str, bool] | None = None,
    search_k: int = 10,
    doc_store_k: int = 12,
    train_env: str = "local_legacy",
    train_session: Any | None = None,
) -> tuple[list[StudentDecisionPoint], list[dict[str, Any]], float, dict[str, Any]]:
    from trim.eval.harmony_runtime import (
        build_continuation_prompt_ids,
        build_first_turn_prompt_ids,
        make_action,
        make_observation,
    )
    from trim.eval.harness1_metrics import EpisodeTiming, episode_quality_metrics, timed_section
    harness_mask = resolved_rollout_mask(
        component_id, harness_mask=harness_mask, teacher_mode=teacher_mode
    )
    g = is_harness_g(mask=harness_mask, component_ids=component_id)
    if g:
        from trim.eval.harness_g_env import execute_tool, new_state, wm_text
        from trim.eval.harness_g_runtime import build_prompt_ids as build_g_prompt_ids
    else:
        from trim.eval.local_search_env import execute_tool, wm_text
        from trim.training.upstream_train_env import (
            apply_train_action,
            is_upstream_state,
            new_state_fn,
            wm_text_for_train_state,
        )
    import torch

    query = str(row["query"])
    qid = str(row["query_id"])
    gold_ids = [str(x) for x in (row.get("gold_docids") or row.get("evidence_docids") or [])]
    store = doc_store_for_row(row, searcher, k=doc_store_k)
    if g:
        st = new_state(query, store, harness_mask=harness_mask)
    else:
        st = new_state_fn(
            train_env=train_env,
            harness_mask=harness_mask,
            session=train_session,
        )(query, store, qid)
    acts: list[tuple[Any, Any]] = []
    points: list[StudentDecisionPoint] = []
    rows: list[dict[str, Any]] = []
    valids: list[bool] = []
    actions: list[dict[str, Any]] = []
    names: list[str] = []
    timing = EpisodeTiming()
    for turn in range(max_turns):
        if st.get("ended"):
            break
        with timed_section(timing, "harness"):
            if g:
                pids = build_g_prompt_ids(
                    query,
                    wm_text(st),
                    enc,
                    harness_mask=harness_mask,
                    actions_obs=acts,
                )
            elif enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
                if turn == 0:
                    pids = enc.build_first_turn_prompt_ids(query)
                else:
                    pids = enc.build_continuation_prompt_ids(
                        query,
                        actions_obs=acts,
                        wm_text=wm_text_for_train_state(st),
                    )
            elif turn == 0:
                pids = build_first_turn_prompt_ids(query, enc=enc)
            else:
                pids = build_continuation_prompt_ids(
                    query,
                    actions_obs=acts,
                    wm_text=wm_text_for_train_state(st),
                    enc=enc,
                )
            pre = snap_from_state(qid, st, component_id, harness_mask=harness_mask)
            student_prefix = render_student_prompt(pre, component_id=component_id)
        if teacher_mode:
            action = teacher_action_from_point(
                StudentDecisionPoint(
                    episode_id=f"{qid}_r{rollout_idx}",
                    query_id=qid,
                    rollout_idx=rollout_idx,
                    turn_id=turn,
                    policy_version=policy_version,
                    pre_action_snapshot=pre,
                    pre_action_snapshot_hash=pre.content_hash(),
                    student_model_input=student_prefix,
                    student_action_tokens=[],
                    student_action_text="",
                    action_tool_names=[],
                    post_action_snapshot=pre,
                    reward=None,
                    structurally_valid=True,
                ),
                component_id,
            )
            valid = True
            text = render_action(action)
            gen = {
                "prompt_ids": pids,
                "action_ids": backend.encode(text),
                "text": text,
                "prompt_text": "teacher_full",
            }
        else:
            with timed_section(timing, "model"):
                gen = generate_harmony(
                    backend,
                    query,
                    enc=enc,
                    max_new=max_new,
                    sample=sample,
                    seed=seed + 17 * rollout_idx + turn,
                    prompt_ids=pids,
                )
            with timed_section(timing, "harness"):
                action, valid = parse_generated_action(
                    gen["text"],
                    gen["action_ids"],
                    enc,
                    harness_mask=harness_mask,
                    teacher_mode=teacher_mode,
                    action_map=st.get("action_map"),
                    finish_reason=str(gen.get("finish_reason") or ""),
                )
        with timed_section(timing, "harness"):
            valids.append(valid)
            actions.append(action)
            names.append(str(action.get("name")))
            exec_ok = True
            if g:
                st, obs, exec_ok = execute_tool(
                    st,
                    action.get("name") if valid else None,
                    action.get("arguments"),
                    searcher=searcher,
                    search_k=search_k,
                )
                if valid and not exec_ok:
                    valid = False
            else:
                st, obs, _ok = apply_train_action(
                    st,
                    action,
                    valid,
                    searcher=searcher,
                    search_k=search_k,
                    mods=st.get("_upstream_mods"),
                    execute_local=execute_tool,
                )
            if valid and exec_ok:
                try:
                    acts.append((make_action(action["name"], action.get("arguments") or {}), make_observation(obs)))
                except Exception:
                    pass
        action_ids = list(gen["action_ids"]) or backend.encode(render_action(action) if valid else "to=unknown\n{}\n")
        prompt_ids = list(gen["prompt_ids"])
        from trim.eval.harmony_runtime import fit_prompt_ids_to_context

        effective_prompt_ids = fit_prompt_ids_to_context(
            list(prompt_ids),
            max_model_len=int(getattr(enc, "max_model_len", 8192) or 8192),
            max_new_tokens=int(max_new),
        )
        with timed_section(timing, "model"):
            with torch.no_grad():
                old_lp = backend._teacher_forced_logprobs(
                    effective_prompt_ids, action_ids, require_grad=False
                )
        token_logprobs = [float(x) for x in old_lp.detach().cpu().tolist()] if old_lp.numel() else []
        old_mean = float(old_lp.mean().item()) if old_lp.numel() else 0.0
        prompt_ids = effective_prompt_ids
        with timed_section(timing, "harness"):
            post = snap_from_state(qid, st, component_id, harness_mask=harness_mask)
            teacher_prompt_ids: list[int] = []
            if enc is not None and not teacher_mode:
                from trim.training.opd_prompt_encoding import encode_teacher_rollout_style_prompt

                teacher_st = dict(st)
                teacher_st["harness_mask"] = teacher_mask_for(component_id)
                if is_upstream_state(teacher_st):
                    teacher_wm = wm_text_for_train_state(teacher_st)
                else:
                    from trim.eval.local_search_env import wm_text as local_wm_text

                    teacher_wm = local_wm_text(teacher_st)
                teacher_prompt_ids, _ = encode_teacher_rollout_style_prompt(
                    enc,
                    query,
                    acts=acts,
                    wm_text=teacher_wm,
                )
            points.append(
                StudentDecisionPoint(
                    episode_id=f"{qid}_r{rollout_idx}",
                    query_id=qid,
                    rollout_idx=rollout_idx,
                    turn_id=turn,
                    policy_version=policy_version,
                    pre_action_snapshot=pre,
                    pre_action_snapshot_hash=pre.content_hash(),
                    student_model_input=student_prefix,
                    student_action_tokens=action_ids,
                    student_action_text=gen["text"],
                    action_tool_names=[action.get("name") or ""],
                    post_action_snapshot=post,
                    reward=None,
                    structurally_valid=valid,
                    student_prompt_token_ids=list(prompt_ids),
                    teacher_prompt_token_ids=list(teacher_prompt_ids),
                )
            )
        rows.append(
            {
                "query_id": qid,
                "episode_id": f"{qid}_r{rollout_idx}",
                "rollout_idx": rollout_idx,
                "prompt": gen.get("prompt_text") or student_prefix,
                "prompt_ids": prompt_ids,
                "effective_prompt_ids": list(prompt_ids),
                "action_text": gen["text"],
                "action_ids": action_ids,
                "token_logprobs": token_logprobs,
                "action_mask": [1] * len(action_ids),
                "logprob_old": old_mean,
                "logprob_provenance": "hf_teacher_forced",
                "n_tokens": len(action_ids),
                "policy_version": policy_version,
                "valid": valid,
                "turn_id": turn,
            }
        )
    reward = terminal_reward(st, query=query, gold_ids=gold_ids, valids=valids, actions=actions)
    for point in points:
        point.reward = reward
    for row_i in rows:
        row_i["reward"] = reward
    stats = episode_quality_metrics(
        st,
        row,
        tool_names=names,
        valids=valids,
        reward=reward,
        max_turns=max_turns,
        timing=timing.snapshot(),
        actions=actions,
    )
    stats.update(
        {
            "names": names,
            "generated_actions": actions,
            "tool_cost": float(st.get("n_tool_calls") or 0),
        }
    )
    return points, rows, reward, stats


def rollout_group(backend, *, row, component_id, group_size, max_turns, max_new, policy_version, seed, sample, enc, searcher=None, teacher_mode=False, harness_mask=None) -> HybridRolloutGroup:
    points: list[StudentDecisionPoint] = []
    rewards: list[float] = []
    rl_rows: list[dict[str, Any]] = []
    tool_seqs: list[list[str]] = []
    harness_mask = resolved_rollout_mask(
        component_id, harness_mask=harness_mask, teacher_mode=teacher_mode
    )
    for g in range(group_size):
        ep_points, ep_rows, reward, stats = one_episode(
            backend,
            row=row,
            component_id=component_id,
            max_turns=max_turns,
            max_new=max_new,
            policy_version=policy_version,
            seed=seed,
            sample=sample,
            enc=enc,
            rollout_idx=g,
            searcher=searcher,
            teacher_mode=teacher_mode,
            harness_mask=harness_mask,
        )
        points.extend(ep_points)
        rl_rows.extend(ep_rows)
        rewards.append(reward)
        tool_seqs.append(list(stats["names"]))
    for rec in rl_rows:
        rec.setdefault("episode_id", f"{row['query_id']}_r{rollout_idx}")
        rec.setdefault("rollout_idx", rollout_idx)
    adv = episode_relative_advantages(rl_rows)
    for rec, a in zip(rl_rows, adv):
        rec["advantage"] = a
    return HybridRolloutGroup(
        query_id=str(row["query_id"]),
        policy_version=policy_version,
        trajectory_group={"rl_rows": rl_rows, "query": row.get("query"), "tool_seqs": tool_seqs},
        decision_points=points,
        terminal_rewards=rewards,
        metadata={"n_rl_rows": len(rl_rows), "reward_spread": max(rewards) - min(rewards) if rewards else 0.0},
    )


def group_stats(groups: list[HybridRolloutGroup]) -> dict[str, Any]:
    n_const = sum(1 for g in groups if len(set(round(r, 6) for r in g.terminal_rewards)) <= 1)
    return {
        "n_groups": len(groups),
        "n_constant_reward_groups": n_const,
        "n_variable_reward_groups": len(groups) - n_const,
        "reward_mean": sum(r for g in groups for r in g.terminal_rewards)
        / max(1, sum(len(g.terminal_rewards) for g in groups)),
        "n_decision_points": sum(len(g.decision_points) for g in groups),
    }


async def train_cell(
    *,
    name: str,
    backend,
    groups: list[HybridRolloutGroup],
    lambda_opd: float,
    train_steps: int,
    policy_version: str,
    opd_states_per_trajectory: int,
    component_id: str,
    teacher_fn: TeacherFn | None,
    opd_loss: str = "sr_opd_ce",
    opd_gate_beta: float = SCAPE_RL_OPD_GATE_BETA,
    groups_per_step: int = HF_DEFAULT_GROUPS_PER_STEP,
    micro_batch_size: int = HF_DEFAULT_MICRO_BATCH,
    heartbeat_every: int = HF_DEFAULT_HEARTBEAT_EVERY,
    skip_group_sampling: bool = False,
    global_optimizer_step: int = 0,
    base_seed: int = 0,
    model_enc: Any | None = None,
) -> dict[str, Any]:
    if name in {"teacher", "before"} or train_steps <= 0:
        return {
            "update_type": "eval_only",
            "n_optimizer_steps": 0,
            "n_rl_forward_backward": 0,
            "n_opd_forward_backward": 0,
            "skipped_teacher": True,
        }
    max_full = int(getattr(backend, "max_full_tokens", 0) or getattr(model_enc, "max_model_len", 8192) or 8192)
    client = HFDebugTrainingClient(
        backend,
        micro_batch_size=micro_batch_size,
        heartbeat_every=heartbeat_every,
        max_full_tokens=max_full,
    )
    teacher = None if lambda_opd <= 0 else teacher_fn
    metrics_acc: list[dict[str, Any]] = []
    last_batch_stats: dict[str, Any] = {}
    step_samples: list[dict[str, Any]] = []
    pool_n = len(groups)
    for step in range(train_steps):
        step_seed = int(base_seed) + int(global_optimizer_step) + int(step)
        if skip_group_sampling or groups_per_step <= 0 or groups_per_step >= len(groups):
            step_groups, sample_meta = groups, {
                "sampled": False,
                "n_groups": len(groups),
                "n_pool": len(groups),
                "query_ids": [str(g.query_id) for g in groups],
                "seed": step_seed,
                "preselected": True,
            }
        else:
            step_groups, sample_meta = sample_groups_for_step(
                groups, groups_per_step, seed=step_seed
            )
        rl_by_q = {
            g.query_id: list((g.trajectory_group or {}).get("rl_rows") or []) for g in step_groups
        }
        client._step_tag = int(global_optimizer_step) + step + 1
        t0 = time.perf_counter()
        log_train(
            "optim_step_start",
            step=step + 1,
            n_steps=int(train_steps),
            cell=name,
            n_pool_groups=pool_n,
            n_step_groups=int(sample_meta["n_groups"]),
            sampled=bool(sample_meta["sampled"]),
            query_ids=sample_meta.get("query_ids") or [],
            micro_batch_size=int(micro_batch_size),
        )
        batch = prepare_hybrid_batch(
            groups=step_groups,
            rl_datums_by_query=rl_by_q,
            policy_version=policy_version,
            lambda_opd=lambda_opd,
            component_id=component_id,
            teacher_event_fn=teacher,
            encode_fn=backend.encode,
            model_enc=model_enc,
            opd_states_per_trajectory=opd_states_per_trajectory,
            seed=step,
            remove_constant_reward_groups=True,
            include_format_errors=uses_sampled_opd(opd_loss),
            opd_loss=opd_loss,
            opd_gate_beta=opd_gate_beta,
        )
        last_batch_stats = dict(batch.projection_stats)
        last_batch_stats["step_sample"] = sample_meta
        if name == "pure_opd":
            rl_use, opd_use = [], batch.opd_datums
        elif name == "rl":
            rl_use, opd_use = batch.rl_datums, []
        else:
            rl_use, opd_use = batch.rl_datums, batch.opd_datums
        log_train(
            "optim_step_datums",
            step=step + 1,
            n_rl_datums=len(rl_use),
            n_opd_datums=len(opd_use),
            n_rl_tokens=int(batch.n_rl_tokens),
            n_opd_tokens=int(batch.n_opd_tokens),
        )
        if not rl_use and not opd_use:
            md = {
                "update_type": "skipped_no_signal",
                "n_optimizer_steps": 0,
                "n_rl_forward_backward": 0,
                "n_opd_forward_backward": 0,
                "skipped_no_signal": True,
            }
        else:
            m = await hybrid_train_substep(
                training_client=client,
                rl_datums=rl_use,
                opd_datums=opd_use,
                rl_loss_fn="cispo",
                rl_loss_fn_config={"clip_low_threshold": 0, "clip_high_threshold": 5},
                lambda_opd=lambda_opd,
                adam_params={},
                policy_version=policy_version,
                projection_coverage=float(batch.projection_stats.get("projection_coverage") or 0.0),
                reject_rate=float(batch.projection_stats.get("reject_rate") or 0.0),
                opd_loss=opd_loss,
            )
            md = m.to_dict()
        elapsed = round(time.perf_counter() - t0, 3)
        md["elapsed_s"] = elapsed
        md["step_sample"] = sample_meta
        metrics_acc.append(md)
        step_samples.append(sample_meta)
        log_train(
            "optim_step_done",
            step=step + 1,
            n_steps=int(train_steps),
            elapsed_s=elapsed,
            n_rl_datums=len(rl_use),
            n_opd_datums=len(opd_use),
            n_rl_tokens=int(batch.n_rl_tokens),
            n_opd_tokens=int(batch.n_opd_tokens),
            update_type=md.get("update_type"),
        )
    from trim.training.tinker_rl_opd_trainer import HybridLoopState

    loop = HybridLoopState(policy_version=policy_version)
    for md in metrics_acc:
        if int(md.get("n_optimizer_steps") or 0) > 0:
            loop.bump_after_update()
    return {
        "call_log": list(client.calls),
        "n_optimizer_steps": sum(1 for c in client.calls if c[0] == "opt"),
        "n_rl_forward_backward": sum(1 for c in client.calls if c[:2] == ("fb", "cispo")),
        "n_opd_forward_backward": sum(
            1
            for c in client.calls
            if c[:2] in {("fb", "cross_entropy"), ("fb", "sampled_gap"), ("fb", "reverse_kl")}
        ),
        "projection_stats": last_batch_stats,
        "substeps": metrics_acc,
        "step_samples": step_samples,
        "train_groups_per_step": int(groups_per_step),
        "train_micro_batch_size": int(micro_batch_size),
        "backend": HFDebugTrainingClient.backend_name,
        "policy_version_start": policy_version,
        "policy_version_end": loop.policy_version if metrics_acc else policy_version,
    }


def eval_closed_loop(
    backend,
    rows: list[dict[str, Any]],
    *,
    component_id,
    max_new,
    max_turns,
    seed,
    enc,
    searcher,
    generate_batch: Callable[[Any], Any] | None = None,
    teacher_mode: bool = False,
    harness_mask: dict[str, bool] | None = None,
    sample: bool | None = None,
    temperature: float = 0.0,
    search_k: int | None = None,
    doc_store_k: int | None = None,
    query_batch_size: int | None = None,
    doc_store_workers: int | None = None,
    primary_split: str = "official_test",
    train_env: str = "local_legacy",
    train_session: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from trim.eval.eval_defaults import (
        HARNESS1_EVAL_DOC_STORE_K,
        HARNESS1_EVAL_SEARCH_K,
    )

    if sample is None:
        sample = float(temperature) > 0.0
    search_k = HARNESS1_EVAL_SEARCH_K if search_k is None else int(search_k)
    doc_store_k = HARNESS1_EVAL_DOC_STORE_K if doc_store_k is None else int(doc_store_k)
    harness_mask = resolved_rollout_mask(
        component_id, harness_mask=harness_mask, teacher_mode=teacher_mode
    )
    g_eval = is_harness_g(mask=harness_mask, component_ids=component_id)
    runtime_audit = None
    if not g_eval:
        from trim.eval.runtime_effect_audit import audit_mask_wiring, merge_audits, summarize_live_effects

        runtime_audit = merge_audits(audit_mask_wiring(harness_mask))
        if not runtime_audit.get("wiring", {}).get("pass"):
            raise RuntimeError(runtime_audit.get("summary") or "mask wiring probe failed")
    if generate_batch is not None:
        from trim.training.batched_env_rollout import rollout_queries_batched, traces_from_groups

        groups = rollout_queries_batched(
            generate_batch,
            rows,
            component_id=component_id,
            group_size=1,
            max_turns=max_turns,
            max_new=max_new,
            policy_version="eval",
            seed=seed,
            sample=bool(sample),
            enc=enc,
            searcher=searcher,
            teacher_mode=teacher_mode,
            harness_mask=harness_mask,
            temperature=float(temperature) if sample else 0.0,
            search_k=search_k,
            doc_store_k=doc_store_k,
            query_batch_size=query_batch_size,
            doc_store_workers=8 if doc_store_workers is None else int(doc_store_workers),
            train_env=train_env,
            train_session=train_session,
        )
        traces, leak = traces_from_groups(groups, rows, searcher=searcher)
        if runtime_audit is not None:
            from trim.eval.runtime_effect_audit import merge_audits, summarize_live_effects

            live = summarize_live_effects(traces, harness_mask)
            runtime_audit = merge_audits(runtime_audit.get("wiring") or runtime_audit, live)
            if not live.get("pass"):
                raise RuntimeError("; ".join(live.get("failures") or ["live effect gate failed"]))
        retrieval_name = searcher.name if searcher is not None else "none"
        split = split_summaries(
            traces,
            setting="closed_loop",
            retrieval_name=retrieval_name,
            eval_rows=rows,
            harness_g=g_eval,
        )
        official = pack_closed_loop_summary(
            split,
            leak=leak,
            n_rows=len(rows),
            primary_split=primary_split,
            extra={
                "max_turns": int(max_turns),
                "max_new_tokens": int(max_new),
                "temperature": float(temperature),
                "search_k": int(search_k),
                "doc_store_k": int(doc_store_k),
                "sample": bool(sample),
                "teacher_leak_count": int(leak),
                "runtime_effect_audit": runtime_audit,
                "claim_usable_for_full_vs_zero": bool((runtime_audit or {}).get("claim_usable_for_full_vs_zero")),
                "eval_harness": "Harness-G" if g_eval else "H_min",
            },
        )
        return official, traces
    traces: list[dict[str, Any]] = []
    leak = 0
    for i, row in enumerate(rows):
        _points, _rl, reward, stats = one_episode(
            backend,
            row=row,
            component_id=component_id,
            max_turns=max_turns,
            max_new=max_new,
            policy_version="eval",
            seed=seed + i,
            sample=bool(sample),
            enc=enc,
            rollout_idx=0,
            searcher=searcher,
            harness_mask=harness_mask,
            search_k=search_k,
            doc_store_k=doc_store_k,
        )
        prefix = render_student_prompt(
            snap_from_state(
                row["query_id"],
                {"query": row["query"], "doc_store": {}, "curated": {}, "pool": {}},
                component_id,
                harness_mask=harness_mask,
            ),
            component_id=component_id,
        )
        if "compressed_teacher_view" in prefix or "VERIFY_RESULT_SECRET" in prefix:
            leak += 1
        search_q = str(row.get("query") or "")
        sm = search_metrics(searcher, search_q, list(row.get("evidence_docids") or [])) if searcher is not None else {}
        from trim.eval.harness1_metrics import trace_fields

        traces.append(
            {
                "query_id": row["query_id"],
                "tool_names": list(stats["names"]),
                **trace_fields(stats),
                **sm,
            }
        )
    retrieval_name = searcher.name if searcher is not None else "none"
    if runtime_audit is not None:
        from trim.eval.runtime_effect_audit import merge_audits, summarize_live_effects

        live = summarize_live_effects(traces, harness_mask)
        runtime_audit = merge_audits(runtime_audit.get("wiring") or runtime_audit, live)
        if not live.get("pass"):
            raise RuntimeError("; ".join(live.get("failures") or ["live effect gate failed"]))
    split = split_summaries(
        traces,
        setting="closed_loop",
        retrieval_name=retrieval_name,
        eval_rows=rows,
        harness_g=g_eval,
    )
    official = pack_closed_loop_summary(
        split,
        leak=leak,
        n_rows=len(rows),
        primary_split=primary_split,
        extra={
            "max_turns": int(max_turns),
            "max_new_tokens": int(max_new),
            "temperature": float(temperature),
            "search_k": int(search_k),
            "doc_store_k": int(doc_store_k),
            "sample": bool(sample),
            "teacher_leak_count": int(leak),
            "runtime_effect_audit": runtime_audit,
            "claim_usable_for_full_vs_zero": bool((runtime_audit or {}).get("claim_usable_for_full_vs_zero")),
        },
    )
    return official, traces


def resolve_queries(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    custom_train = getattr(args, "query_manifest", None)
    n_queries = getattr(args, "n_queries", None)
    if uses_sec_train_data(args):
        train_rows, train_src = load_sec_rl_queries(
            getattr(args, "rl_data", None) or default_sec_rl_data(),
            n_queries=n_queries,
            query_file=Path(custom_train) if custom_train else None,
            corpus_root=getattr(args, "sec_corpus_root", None) or default_sec_corpus_root(),
        )
        if not getattr(args, "validate_only", False) and not getattr(args, "dry_run", False):
            train_src["doc_store"] = attach_sec_doc_stores(
                train_rows,
                corpus_root=getattr(args, "sec_corpus_root", None) or default_sec_corpus_root(),
            )
        eval_rows, eval_src = load_bcplus_830_full()
        overlap = overlap_ids(train_rows, eval_rows)
        train_meta = {
            **train_src,
            "score_split": SCORE_SPLIT_830,
        }
        eval_meta = {
            **eval_src,
            "official_test_count": BCPLUS_TEST,
            "official_test_expected": BCPLUS_TEST,
            "eval_count": len(eval_rows),
        }
        states_path = getattr(args, "train_states", None)
        frozen_limit = None if getattr(args, "n_train_states", None) in {None, 0} else int(args.n_train_states)
        if not states_path:
            frozen_points, frozen_meta = [], {"found": False, "path": None, "n_states": 0, "source": "current_on_policy_rl_rollout"}
        else:
            frozen_points, frozen_meta = load_train_states(
                Path(states_path),
                component_id=getattr(args, "component", "sentence_compress"),
                limit=frozen_limit,
            )
        if frozen_points:
            by_q = {p.query_id: True for p in frozen_points}
            for row in train_rows:
                store = doc_store_from_points(frozen_points, row["query_id"])
                if store:
                    row["frozen_doc_store"] = store
            frozen_meta["n_train_rows_with_docs"] = sum(1 for r in train_rows if r.get("frozen_doc_store"))
            frozen_meta["n_frozen_query_overlap_train"] = sum(1 for r in train_rows if r["query_id"] in by_q)
        return (
            train_rows,
            eval_rows,
            {"train": train_meta, "eval": eval_meta, "overlap": overlap, "frozen_states": frozen_meta},
            frozen_points,
        )

    train_rows, test_rows, split_meta = load_bcplus_830_split(n_train=None)
    test_ids = {r["query_id"] for r in test_rows}
    custom_train = getattr(args, "query_manifest", None)
    if custom_train:
        extra, custom_meta = load_train_queries(
            manifest=Path(custom_train),
            n_queries=None,
            exclude_eval_ids=test_ids,
        )
        allowed = {r["query_id"] for r in train_rows}
        train_rows = [r for r in extra if r["query_id"] in allowed]
        split_meta = dict(split_meta)
        split_meta["custom_train_manifest"] = custom_meta
    n_queries = getattr(args, "n_queries", None)
    if n_queries not in {None, 0} and int(n_queries) < len(train_rows):
        train_rows = train_rows[: int(n_queries)]
    eval_manifest = getattr(args, "eval_manifest", None)
    if eval_manifest:
        from trim.eval.official_query_pool import load_query_manifest

        eval_rows = attach_bcp_fields(load_query_manifest(Path(eval_manifest)))
        eval_rows = [r for r in eval_rows if r["query_id"] in test_ids]
        for rec in eval_rows:
            rec["official_split"] = "test"
        if not eval_rows:
            eval_rows = test_rows
    else:
        eval_rows = test_rows
    train_rows = attach_bcp_fields(train_rows)
    overlap = overlap_ids(train_rows, eval_rows)
    if overlap:
        raise RuntimeError(f"train/eval query overlap: {overlap[:8]}")
    eval_meta = {
        **split_meta,
        "query_count": len(eval_rows),
        "official_test_count": len(eval_rows),
        "official_test_expected": BCPLUS_TEST,
        "score_split": SCORE_SPLIT_166,
    }
    train_meta = {
        **split_meta,
        "query_count": len(train_rows),
        "using_full_train_split": len(train_rows) == BCPLUS_TRAIN,
    }
    states_path = getattr(args, "train_states", None)
    frozen_limit = None if getattr(args, "n_train_states", None) in {None, 0} else int(args.n_train_states)
    if not states_path:
        frozen_points, frozen_meta = [], {"found": False, "path": None, "n_states": 0, "source": "current_on_policy_rl_rollout"}
    else:
        frozen_points, frozen_meta = load_train_states(
            Path(states_path),
            component_id=getattr(args, "component", "sentence_compress"),
            limit=frozen_limit,
        )
    if frozen_points:
        by_q = {p.query_id: True for p in frozen_points}
        for row in train_rows:
            store = doc_store_from_points(frozen_points, row["query_id"])
            if store:
                row["frozen_doc_store"] = store
        frozen_meta["n_train_rows_with_docs"] = sum(1 for r in train_rows if r.get("frozen_doc_store"))
        frozen_meta["n_frozen_query_overlap_train"] = sum(1 for r in train_rows if r["query_id"] in by_q)
    return (
        train_rows,
        eval_rows,
        {"train": train_meta, "eval": eval_meta, "overlap": overlap, "frozen_states": frozen_meta},
        frozen_points,
    )


def validate_wiring(args: argparse.Namespace) -> dict[str, Any]:
    from trim.state.snapshot import EnvironmentSnapshot
    from trim.training.opd_dataset import project_and_materialize
    from trim.training.opd_projection import StudentActionSpaceProjector

    teacher_fn = teacher_for(
        args.component,
        harness=getattr(args, "harness", None),
        teacher_kind=str(getattr(args, "teacher_kind", "upstream") or "upstream"),
    )
    if teacher_fn is None:
        raise SystemExit(f"no teacher registered for component={args.component}")
    train_rows, eval_rows, pool_meta, frozen_points = resolve_queries(args)
    harness = getattr(args, "harness", None) or infer_harness_from_ids(args.component)
    mask = student_mask_for(args.component, harness=harness)
    wm = {
        "query": train_rows[0]["query"],
        "documents": [{"id": "d_long", "text": ("Long noisy passage. " * 40) + train_rows[0]["query"]}],
        "curated_ids": [],
    }
    if is_harness_g(harness):
        from trim.eval.harness_g_env import execute_tool as g_execute
        from trim.eval.harness_g_env import new_state as g_new_state

        g_st = g_new_state(
            train_rows[0]["query"],
            {"d_long": wm["documents"][0]},
            harness_mask=mask,
        )
        g_st, _, _ = g_execute(g_st, "init", {})
        wm.update(
            {
                "visible_sids": list(g_st.get("visible_sids") or []),
                "selected_sids": list(g_st.get("selected_sids") or []),
                "sentences": g_st.get("sentences") or {},
                "entities": g_st.get("entities") or {},
                "frontier_eids": list(g_st.get("frontier_eids") or []),
                "action_map": g_st.get("action_map") or {},
                "initialized": True,
                "accessible_doc_ids": list(g_st.get("visible_sids") or []) + list((g_st.get("entities") or {}).keys()),
            }
        )
    events = teacher_events_from_wm_for(args.component, wm)
    snap = capture_snapshot(
        query_id=train_rows[0]["query_id"],
        step=0,
        harness_mask=mask,
        working_memory=wm,
        metadata={"component_id": args.component, "harness": "Harness-G" if is_harness_g(harness) else "Harness-1"},
    )
    projection, steps = project_and_materialize(
        student_snapshot=snap,
        teacher_events=events,
        student_mask=snap.harness_mask,
        component_id=args.component,
        projector=StudentActionSpaceProjector(),
    )
    leaked = any("compressed_teacher_view" in (s.prompt_reduced or "") for s in steps)
    return {
        "ok": True,
        "component": args.component,
        "harness": harness,
        "component_ids": component_ids_of(args.component, harness=harness),
        "teacher_registered": True,
        "n_train_queries": len(train_rows),
        "n_eval_queries": len(eval_rows),
        "eval_is_official_384": False,
        "using_full_train_split": bool((pool_meta.get("train") or {}).get("using_full_train_split", len(train_rows) == BCPLUS_TRAIN)),
        "official_test_count": int(pool_meta["eval"].get("official_test_count") or 0),
        "official_test_is_166": int(pool_meta["eval"].get("official_test_count") or 0) == BCPLUS_TEST
        and not uses_bcplus_830_eval(args),
        "eval_is_bcplus_830": uses_bcplus_830_eval(args) and len(eval_rows) == BCPLUS_TOTAL,
        "train_pool": (pool_meta.get("train") or {}).get("pool_contract")
        or (SEC_TRAIN_POOL_NAME if uses_sec_train_data(args) else "bcplus_train_664"),
        "score_split": SCORE_SPLIT_830 if uses_bcplus_830_eval(args) else SCORE_SPLIT_166,
        "official_test_is_76": False,
        "train_states": pool_meta.get("frozen_states") or {},
        "n_frozen_states": len(frozen_points),
        "projection_kind": projection.kind.value,
        "n_projected_steps": len(steps),
        "teacher_leak_in_student_prefix": leaked,
        "pool": pool_meta,
        "snapshot_type": type(snap).__name__,
        "environment_snapshot": EnvironmentSnapshot.__name__,
    }


def uses_vllm(args: argparse.Namespace) -> bool:
    return str(getattr(args, "rollout_backend", "vllm") or "vllm") == "vllm"


def uses_scheme_a(args: argparse.Namespace) -> bool:
    return uses_vllm(args) and str(getattr(args, "gpu_schedule", "scheme_a") or "scheme_a") == "scheme_a"


def open_train_retrieval(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]] | None = None,
) -> RetrievalBackend:
    """Train-time searcher. SEC train data uses the SEC parquet/BM25 corpus."""
    smoke = bool(getattr(args, "smoke", False))
    if uses_sec_train_data(args):
        texts: dict[str, str] = {}
        for row in train_rows or []:
            for did, rec in (row.get("seed_doc_store") or {}).items():
                if isinstance(rec, dict) and rec.get("text"):
                    texts[str(did)] = str(rec["text"])
        root = getattr(args, "sec_corpus_root", None) or default_sec_corpus_root()
        backend = open_sec_retrieval(root, texts=texts or None)
        if not smoke and backend.name != "sec_pyserini":
            index = corpus_bm25_index(root)
            raise RuntimeError(
                f"SEC training requires a local Lucene BM25 index at {index}. "
                "Build it with: PYTHONPATH=TRIM python TRIM/scripts/build_sec_bm25_index.py"
            )
        return backend
    return open_retrieval(formal=not smoke)


def open_eval_retrieval(args: argparse.Namespace) -> RetrievalBackend:
    """Eval always searches BrowseComp-Plus, including SEC-train runs scored on BC+ 830."""
    return open_retrieval(formal=not bool(getattr(args, "smoke", False)))


def train_device_map_for(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "train_device_map", "") or "")
    if explicit:
        return explicit
    if uses_scheme_a(args):
        return "auto"
    return f"cuda:{int(getattr(args, 'gpu', 0))}"


def load_hf_backend(args: argparse.Namespace, device_map: str, *, adapter_dir: str | None = None):
    from trim.training.hf_tool_opd import ScapeHFToolOPD
    from trim.training.vllm_hybrid import load_adapter_weights

    model_src = args.sft_adapter if args.sft_adapter and Path(args.sft_adapter).exists() else args.base_model
    backend = ScapeHFToolOPD(
        model_path=model_src,
        base_model_override=args.base_model,
        device_map=device_map,
        learning_rate=1e-5,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    )
    backend.max_full_tokens = int(getattr(args, "max_model_len", 8192) or 8192)
    if adapter_dir:
        load_adapter_weights(backend, adapter_dir)
    return backend


def merge_train_stats(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {
            "update_type": "eval_only",
            "n_optimizer_steps": 0,
            "n_rl_forward_backward": 0,
            "n_opd_forward_backward": 0,
            "skipped_teacher": True,
        }
    last = parts[-1]
    return {
        **last,
        "call_log": [c for p in parts for c in (p.get("call_log") or [])],
        "n_optimizer_steps": sum(int(p.get("n_optimizer_steps") or 0) for p in parts),
        "n_rl_forward_backward": sum(int(p.get("n_rl_forward_backward") or 0) for p in parts),
        "n_opd_forward_backward": sum(int(p.get("n_opd_forward_backward") or 0) for p in parts),
        "substeps": [s for p in parts for s in (p.get("substeps") or [])],
        "n_on_policy_rollouts": len(parts),
        "policy_version_start": parts[0].get("policy_version_start"),
        "policy_version_end": last.get("policy_version_end"),
    }


def run_four_cell(args: argparse.Namespace) -> dict[str, Any]:
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive

    keepalive = acquire_keepalive()
    try:
        return _run_four_cell_body(args, keepalive)
    finally:
        release_keepalive()


def _run_four_cell_body(args: argparse.Namespace, keepalive) -> dict[str, Any]:
    import torch
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.batched_env_rollout import rollout_queries_batched
    from trim.training.tinker_rl_opd_trainer import HybridLoopState
    from trim.training.vllm_hybrid import (
        HFGenerateClient,
        SchemeARuntime,
        VLLMGenerateClient,
        default_tensor_parallel_size,
        load_adapter_weights,
        materialize_vllm_base,
        wait_gpus_quiet,
        _release_cuda,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_only = bool(getattr(args, "train_only", False))
    log_tag = "train" if train_only else "four_cell"
    from trim.eval.runtime_effect_audit import audit_train_runtime_or_raise

    train_audit = audit_train_runtime_or_raise(args, out=out)
    if not train_audit.get("skipped"):
        print(
            f"[{log_tag}] runtime wiring pass student_on={train_audit.get('student_n_on')} "
            f"teacher_on={train_audit.get('teacher_n_on')} "
            f"opd_steps={train_audit.get('opd_n_projected_steps')}",
            flush=True,
        )
    train_rows, eval_rows, pool_meta, frozen_points = resolve_queries(args)
    train_searcher = open_train_retrieval(args, train_rows)
    eval_searcher = None if train_only else open_eval_retrieval(args)
    from trim.training.upstream_train_env import TRAIN_ENV_UPSTREAM, canonical_train_env, open_upstream_train_session

    train_env = canonical_train_env(getattr(args, "train_env", TRAIN_ENV_UPSTREAM))
    train_session = None
    if train_env == TRAIN_ENV_UPSTREAM and not is_harness_g(
        harness=getattr(args, "harness", None), component_ids=getattr(args, "component", None)
    ):
        from trim.upstream_harness1.retrieval import retrieval_from_args

        try:
            train_session = open_upstream_train_session(
                retrieval=retrieval_from_args(args),
                harness_mask=teacher_mask_for(
                    args.component, harness=getattr(args, "harness", None)
                ),
                max_turns=int(args.max_turns),
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "train_env=upstream failed to open original SlidingWindowSearchEnv: "
                f"{exc}\nChroma backend needs --retrieval-backend upstream and cloud/local "
                "Chroma credentials; Lucene backend needs --retrieval-backend local_bm25 "
                "with --index-path and --corpus-path/--docstore-path. "
                "This does not fall back to TRIM local_search_env. "
                "Pass --train-env local_legacy only for the named substitute."
            ) from exc
    frozen_groups = groups_from_frozen_points(frozen_points) if frozen_points else []
    vllm_on = uses_vllm(args)
    scheme_a = uses_scheme_a(args)
    device_map = train_device_map_for(args)
    # GPT-OSS Transformers backend currently rejects tensor parallelism; use
    # an explicit CLI TP size when supplied, otherwise retain the vLLM default.
    tp = int(getattr(args, "tensor_parallel_size", None) or default_tensor_parallel_size(None))
    manifest = build_manifest(
        args,
        extra={
            "pool": pool_meta,
            "retrieval": train_searcher.name,
            "eval_retrieval": None if eval_searcher is None else eval_searcher.name,
            "cells": list(
                cells_for_mode(
                    getattr(args, "training_mode", "four_cell"),
                    train_only=train_only,
                )
            ),
            "train_env": str(getattr(args, "train_env", "upstream") or "upstream"),
            "teacher_kind": str(getattr(args, "teacher_kind", "upstream") or "upstream"),
            "updates_per_rollout": int(getattr(args, "updates_per_rollout", 1) or 1),
            "tensor_parallel_size": tp,
            "train_device_map": device_map,
            "resolved_vllm": _resolved_vllm_config(args, tp=tp),
        },
    )
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    teacher_fn = teacher_for(
        args.component,
        harness=getattr(args, "harness", None),
        teacher_kind=str(getattr(args, "teacher_kind", "upstream") or "upstream"),
    )
    if teacher_fn is None:
        raise SystemExit(f"no teacher registered for {args.component}")

    t0 = time.time()
    runtime = SchemeARuntime()
    vllm_base = args.base_model
    if vllm_on:
        keepalive.pause()
        vllm_base = materialize_vllm_base(
            base_model=args.base_model,
            sft_adapter=str(args.sft_adapter or ""),
            cache_dir=out / "vllm_base_merged_sft",
            device_map=device_map,
        )
        wait_gpus_quiet()
        keepalive.resume()

    # Do not load a 20B HF LoRA before the first vLLM rollout. That extra load
    # plus unload leaves SM-Util at 0% for minutes and trips cluster killers.
    # First collect_groups(theta0) uses the vLLM base (no adapter file yet).
    backend = None
    theta0_dir = out / "adapters" / "theta0"
    theta0_saved = {"n": False}
    if not scheme_a:
        print(f"[{log_tag}] init theta0 HF LoRA", flush=True)
        keepalive.pause()
        backend = load_hf_backend(args, device_map)
        theta0_dir.mkdir(parents=True, exist_ok=True)
        backend.save_pretrained(str(theta0_dir))
        theta0_saved["n"] = True
        runtime.attach_hf(backend)

    enc = load_model_encoding(str(getattr(args, "base_model", None) or getattr(args, "model_name", None) or ""))
    setattr(enc, "max_model_len", int(getattr(args, "max_model_len", 8192) or 8192))
    chosen_cells = cells_for_mode(
        getattr(args, "training_mode", "four_cell"),
        train_only=train_only,
    )
    adapter_map: dict[str, str | None] = {}
    adapter_audits: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    eval_summaries: list[dict[str, Any]] = []
    session_i = {"n": 0}
    eval_primary = SCORE_SPLIT_830 if uses_bcplus_830_eval(args) else "official_test"
    if uses_bcplus_830_eval(args):
        ev_rows = list(eval_rows)
    else:
        ev_rows = official_test_subset(eval_rows) if getattr(args, "official_eval", True) else train_rows
    if getattr(args, "n_eval", None):
        ev_rows = ev_rows[: int(args.n_eval)]

    def next_session(tag: str) -> Path:
        session_i["n"] += 1
        path = out / "vllm_sessions" / f"{session_i['n']:03d}_{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def vllm_lora(path: str | None, *, required: bool = False) -> str | None:
        if not path:
            if required:
                raise FileNotFoundError("vLLM rollout requested but no adapter path provided")
            return None
        root = Path(path)
        weight = root / "adapter_model.safetensors"
        if weight.is_file():
            return str(root)
        if required or (root.is_dir() and any(root.iterdir())):
            raise FileNotFoundError(f"adapter weights missing: {weight}")
        return None

    def open_vllm(lora_path: str | None, tag: str) -> VLLMGenerateClient:
        keepalive.pause()
        wait_gpus_quiet()
        model_for_vllm = vllm_base
        has_adapter = bool(lora_path and (Path(lora_path) / "adapter_model.safetensors").is_file())
        vllm_adapter = vllm_lora(lora_path, required=has_adapter)
        digest = policy_digest_record(
            adapter_dir=vllm_adapter,
            base_model=model_for_vllm,
            policy_version=tag,
        )
        client = VLLMGenerateClient(
            model_path=model_for_vllm,
            session_dir=next_session(tag),
            tensor_parallel_size=tp,
            max_model_len=int(getattr(args, "max_model_len", 8192) or 8192),
            lora_path=vllm_adapter,
            gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.90) or 0.90),
            enforce_eager=bool(getattr(args, "enforce_eager", True)),
            python_exe=str(getattr(args, "vllm_python", "") or "") or None,
            startup_timeout_s=3600.0,
            generate_timeout_s=float(getattr(args, "vllm_generate_timeout_s", 3600.0) or 3600.0),
            max_num_seqs=int(getattr(args, "max_num_seqs", 0) or 0) or None,
            extra_env={},
        )
        disable_ar = getattr(args, "vllm_disable_custom_all_reduce", None)
        if disable_ar is not None:
            client._disable_custom_all_reduce = bool(disable_ar)
        runtime.attach_vllm(client)
        print(
            f"[{log_tag}] vLLM start backend=vllm tp={tp} lora={client.lora_path} "
            f"digest={digest.get('policy_digest')} tag={tag}",
            flush=True,
        )
        client.start()
        (client.session_dir / "policy_digest.json").write_text(
            json.dumps(digest, indent=2) + "\n", encoding="utf-8"
        )
        return client

    def close_vllm() -> None:
        runtime.detach_vllm()
        wait_gpus_quiet()
        keepalive.resume()

    optimizer_path = out / "tmp" / "optimizer_latest.pt"

    def ensure_hf(adapter_path: str | None):
        nonlocal backend
        import torch

        keepalive.pause()
        if backend is None:
            adapter_file = Path(adapter_path) / "adapter_model.safetensors" if adapter_path else None
            adapter_ok = bool(adapter_file is not None and adapter_file.is_file())
            backend = runtime.attach_hf(
                load_hf_backend(args, device_map, adapter_dir=adapter_path if adapter_ok else None)
            )
            if backend.optimizer is None:
                backend.optimizer = torch.optim.AdamW(
                    [p for p in backend.model.parameters() if p.requires_grad],
                    lr=float(getattr(backend, "learning_rate", 1e-5) or 1e-5),
                )
            if optimizer_path.is_file():
                load_optimizer_bundle(backend, optimizer_path)
            if not theta0_saved["n"]:
                print(f"[{log_tag}] save theta0 HF LoRA after first attach", flush=True)
                theta0_dir.mkdir(parents=True, exist_ok=True)
                backend.save_pretrained(str(theta0_dir))
                theta0_saved["n"] = True
            return backend
        load_adapter_weights(backend, adapter_path)
        if backend.optimizer is None:
            backend.optimizer = torch.optim.AdamW(
                [p for p in backend.model.parameters() if p.requires_grad],
                lr=float(getattr(backend, "learning_rate", 1e-5) or 1e-5),
            )
            if optimizer_path.is_file():
                load_optimizer_bundle(backend, optimizer_path)
        return backend

    def release_hf() -> None:
        nonlocal backend
        if scheme_a and backend is not None:
            runtime.detach_hf(optimizer_path=optimizer_path)
            backend = None
            wait_gpus_quiet()
            keepalive.resume()

    def collect_groups(
        lora_path: str | None,
        policy_version: str,
        tag: str,
        *,
        sample: bool,
        rows,
        group_size: int,
        teacher_mode: bool = False,
    ):
        rollout_backend = str(getattr(args, "rollout_backend", "vllm") or "vllm").lower()
        rollout_kw = dict(
            component_id=args.component,
            group_size=group_size,
            max_turns=args.max_turns,
            max_new=args.max_new_tokens,
            policy_version=policy_version,
            seed=stable_rollout_seed(int(args.seed), tag),
            sample=sample,
            enc=enc,
            searcher=train_searcher,
            teacher_mode=teacher_mode,
            harness_mask=resolved_rollout_mask(
                args.component,
                harness=getattr(args, "harness", None),
                teacher_mode=teacher_mode,
            ),
            query_batch_size=getattr(args, "rollout_query_batch_size", None),
            doc_store_workers=int(getattr(args, "doc_store_workers", 8) or 8),
            train_env=train_env,
            train_session=train_session,
            rollout_backend=rollout_backend,
        )
        if rollout_backend == "vllm":
            if backend is not None:
                release_hf()
            client = open_vllm(lora_path, tag)
            try:
                return rollout_queries_batched(client.generate_batch, rows, **rollout_kw)
            finally:
                close_vllm()
        if rollout_backend == "hf":
            gen = HFGenerateClient(ensure_hf(lora_path), enc=enc)
            rollout_kw["rollout_backend"] = "hf"
            return rollout_queries_batched(gen.generate_batch, rows, **rollout_kw)
        raise RuntimeError(f"unsupported rollout_backend={rollout_backend!r}")

    def eval_now(lora_path: str | None, tag: str, *, teacher_mode: bool = False):
        eval_kw = dict(
            component_id=args.component,
            max_new=int(getattr(args, "eval_max_new_tokens", args.max_new_tokens)),
            max_turns=int(getattr(args, "eval_max_turns", args.max_turns)),
            seed=args.seed,
            enc=enc,
            searcher=eval_searcher,
            teacher_mode=teacher_mode,
            harness_mask=resolved_rollout_mask(
                args.component,
                harness=getattr(args, "harness", None),
                teacher_mode=teacher_mode,
            ),
            temperature=float(getattr(args, "eval_temperature", 0.0)),
            primary_split=eval_primary,
            query_batch_size=getattr(args, "rollout_query_batch_size", None),
            doc_store_workers=int(getattr(args, "doc_store_workers", 8) or 8),
            train_env=train_env,
            train_session=train_session,
        )
        rollout_backend = str(getattr(args, "rollout_backend", "vllm") or "vllm").lower()
        if rollout_backend == "vllm":
            if backend is not None:
                release_hf()
            client = open_vllm(lora_path, tag)
            try:
                return eval_closed_loop(
                    None,
                    ev_rows,
                    generate_batch=client.generate_batch,
                    **eval_kw,
                )
            finally:
                close_vllm()
        if rollout_backend == "hf":
            gen = HFGenerateClient(ensure_hf(lora_path), enc=enc)
            return eval_closed_loop(
                backend,
                ev_rows,
                generate_batch=gen.generate_batch,
                **eval_kw,
            )
        raise RuntimeError(f"unsupported rollout_backend={rollout_backend!r}")

    def save_and_audit(cell: str, adapter_dir: Path) -> dict[str, Any]:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        if backend is None:
            raise RuntimeError(f"cannot save adapter for {cell}: HF backend is not loaded")
        backend.save_pretrained(str(adapter_dir))
        file_audit = audit_saved_adapter(adapter_dir, cell=cell)
        load_adapter_weights(backend, theta0_dir)
        info = load_adapter_weights(backend, adapter_dir)
        file_audit["reload_path"] = "saved_adapter_state_dict"
        file_audit["unexpected_lora"] = info.get("unexpected_lora") or []
        return file_audit

    for cell in chosen_cells:
        adapter_live = str(theta0_dir)
        loop = HybridLoopState(policy_version="v0")
        print(f"[{log_tag}] cell={cell} phases", flush=True)
        groups = None
        train_parts: list[dict[str, Any]] = []
        use_frozen = cell == "pure_opd" and bool(frozen_groups)
        refresh = bool(getattr(args, "on_policy_refresh", False)) and cell not in {"teacher", "before"}
        if use_frozen:
            refresh = False
        n_train = 0 if cell in {"teacher", "before"} else int(args.train_steps)
        updates_per_rollout = int(getattr(args, "updates_per_rollout", 1) or 1)
        if refresh:
            updates_per_rollout = 1

        if use_frozen:
            groups = frozen_groups
        elif refresh:
            query_sampler = QuerySampler(
                train_rows,
                base_seed=int(args.seed),
                groups_per_step=int(
                    getattr(args, "train_groups_per_step", HF_DEFAULT_GROUPS_PER_STEP)
                    or HF_DEFAULT_GROUPS_PER_STEP
                ),
            )
            metrics_path = out / cell / "metrics.jsonl"
            for step in range(n_train):
                step_rows, sample_meta = query_sampler.sample_for_rollout()
                query_sampler.note_rollout_start()
                print(
                    f"[{log_tag}] cell={cell} on-policy rollout step={step} "
                    f"policy={loop.policy_version} queries={sample_meta.get('query_ids')}",
                    flush=True,
                )
                groups = collect_groups(
                    adapter_live,
                    loop.policy_version,
                    f"{cell}_rollout{step}",
                    sample=True,
                    rows=step_rows,
                    group_size=args.group_size,
                    teacher_mode=cell == "teacher",
                )
                rewards_before = [r for g in groups for r in g.terminal_rewards]
                ensure_hf(adapter_live)
                part = asyncio.run(
                    train_cell(
                        name=cell,
                        backend=backend,
                        groups=groups,
                        lambda_opd=cell_lambda(cell, args.lambda_opd),
                        train_steps=1,
                        policy_version=loop.policy_version,
                        opd_states_per_trajectory=args.opd_states_per_trajectory,
                        component_id=args.component,
                        teacher_fn=teacher_fn,
                        opd_loss=str(getattr(args, "opd_loss", None) or "sr_opd_ce"),
                        opd_gate_beta=float(
                            getattr(args, "opd_gate_beta", SCAPE_RL_OPD_GATE_BETA)
                            or SCAPE_RL_OPD_GATE_BETA
                        ),
                        groups_per_step=int(getattr(args, "train_groups_per_step", HF_DEFAULT_GROUPS_PER_STEP) or 0),
                        micro_batch_size=int(getattr(args, "train_micro_batch_size", HF_DEFAULT_MICRO_BATCH) or HF_DEFAULT_MICRO_BATCH),
                        heartbeat_every=int(getattr(args, "train_heartbeat_every", HF_DEFAULT_HEARTBEAT_EVERY) or HF_DEFAULT_HEARTBEAT_EVERY),
                        skip_group_sampling=True,
                        global_optimizer_step=query_sampler.state.global_optimizer_step,
                        base_seed=int(args.seed),
                        model_enc=enc,
                    )
                )
                rewards_after = [r for g in groups for r in g.terminal_rewards]
                if rewards_before != rewards_after:
                    raise RuntimeError("Teacher shadow mutated RL rewards")
                n_opt = int(part.get("n_optimizer_steps") or 0)
                step_num = query_sampler.state.global_optimizer_step + (1 if n_opt > 0 else 0)
                ckpt_tmp = out / "checkpoints" / cell / f".tmp_step_{step_num:06d}"
                ckpt_final = out / "checkpoints" / cell / f"step_{step_num:06d}"
                adapter_step_dir = ckpt_tmp / "adapter"
                adapter_step_dir.mkdir(parents=True, exist_ok=True)
                backend.save_pretrained(str(adapter_step_dir))
                save_optimizer_bundle(backend, ckpt_tmp / "optimizer.pt")
                save_rng_state(ckpt_tmp / "rng.json")
                (ckpt_tmp / "sampler.json").write_text(
                    json.dumps(query_sampler.state.to_dict(), indent=2) + "\n", encoding="utf-8"
                )
                digest = policy_digest_record(
                    adapter_dir=adapter_step_dir,
                    base_model=vllm_base,
                    policy_version=loop.policy_version,
                )
                step_manifest = {
                    "cell": cell,
                    "step": step_num,
                    "policy_version": loop.policy_version,
                    "sample_meta": sample_meta,
                    "train": part,
                    "policy_digest": digest,
                    "n_optimizer_steps": n_opt,
                }
                append_metrics_jsonl(
                    metrics_path,
                    {
                        "step": step_num,
                        "cell": cell,
                        "policy_version": loop.policy_version,
                        "sample_meta": sample_meta,
                        "n_optimizer_steps": n_opt,
                        "update_type": (part.get("substeps") or [{}])[-1].get("update_type")
                        if part.get("substeps")
                        else part.get("update_type"),
                    },
                )
                if n_opt > 0:
                    publish_step_checkpoint(ckpt_tmp, ckpt_final, manifest=step_manifest)
                    query_sampler.note_update_complete()
                    loop.bump_after_update()
                    adapter_live = str(adapter_step_dir)
                    adapter_dir = out / "adapters" / cell
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    backend.save_pretrained(str(adapter_dir))
                    adapter_live = str(adapter_dir)
                    if step == n_train - 1:
                        adapter_audits.append(save_and_audit(cell, adapter_dir))
                else:
                    import shutil

                    if ckpt_tmp.exists():
                        shutil.rmtree(ckpt_tmp)
                    part["skipped_no_signal"] = True
                train_parts.append(part)
                release_hf()
        else:
            print(f"[{log_tag}] cell={cell} rollout", flush=True)
            groups = collect_groups(
                adapter_live,
                loop.policy_version,
                f"{cell}_rollout",
                sample=True,
                rows=train_rows,
                group_size=args.group_size,
                teacher_mode=cell == "teacher",
            )

        if groups is None:
            raise RuntimeError(f"cell={cell} produced no rollout groups")
        if not refresh and n_train > 0:
            rewards_before = [r for g in groups for r in g.terminal_rewards]
            ensure_hf(adapter_live)
            train_parts.append(
                asyncio.run(
                    train_cell(
                        name=cell,
                        backend=backend,
                        groups=groups,
                        lambda_opd=cell_lambda(cell, args.lambda_opd),
                        train_steps=n_train,
                        policy_version=loop.policy_version,
                        opd_states_per_trajectory=args.opd_states_per_trajectory,
                        component_id=args.component,
                        teacher_fn=teacher_fn,
                        opd_loss=str(getattr(args, "opd_loss", None) or "sr_opd_ce"),
                        opd_gate_beta=float(
                            getattr(args, "opd_gate_beta", SCAPE_RL_OPD_GATE_BETA)
                            or SCAPE_RL_OPD_GATE_BETA
                        ),
                        groups_per_step=int(getattr(args, "train_groups_per_step", HF_DEFAULT_GROUPS_PER_STEP) or 0),
                        micro_batch_size=int(getattr(args, "train_micro_batch_size", HF_DEFAULT_MICRO_BATCH) or HF_DEFAULT_MICRO_BATCH),
                        heartbeat_every=int(getattr(args, "train_heartbeat_every", HF_DEFAULT_HEARTBEAT_EVERY) or HF_DEFAULT_HEARTBEAT_EVERY),
                        model_enc=enc,
                    )
                )
            )
            rewards_after = [r for g in groups for r in g.terminal_rewards]
            if rewards_before != rewards_after:
                raise RuntimeError("Teacher shadow mutated RL rewards")
            if cell != "before":
                adapter_dir = out / "adapters" / cell
                adapter_audits.append(save_and_audit(cell, adapter_dir))
                adapter_map[cell] = str(adapter_dir)
                adapter_live = str(adapter_dir)
            release_hf()
        elif cell in {"teacher", "before"}:
            adapter_map[cell] = None
            adapter_audits.append(
                {"cell": cell, "adapter_dir": None, "reload_ready": True, "exists": False, "reload_path": "theta0_no_adapter"}
            )
        elif refresh:
            adapter_map[cell] = adapter_live
            if not any(a.get("cell") == cell for a in adapter_audits):
                adapter_audits.append(audit_saved_adapter(Path(adapter_live), cell=cell))

        collected = filter_component_states(
            [p for g in groups for p in g.decision_points],
            component_id=args.component,
            require_valid=False,
        )
        write_collected_states(
            collected,
            out / cell / "collected_states.jsonl",
            component_id=args.component,
            extra={
                "cell": cell,
                "n_rollout_points": sum(len(g.decision_points) for g in groups),
                "policy_version": groups[0].policy_version if groups else "v0",
                "on_policy_refresh": refresh,
            },
        )
        gstat = group_stats(groups)
        train_stats = merge_train_stats(train_parts)
        ev: dict[str, Any] = {
            "setting": cell,
            "skipped": True,
            "note": "train_only",
        }
        traces: list[dict[str, Any]] = []
        if not train_only:
            print(f"[{log_tag}] cell={cell} eval", flush=True)
            ev, traces = eval_now(
                str(theta0_dir) if cell == "before" else adapter_live,
                f"{cell}_eval",
                teacher_mode=cell == "teacher",
            )
            ev["setting"] = cell
            ev["reported_split"] = "official_test"
        cell_dir = out / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        if traces:
            with (cell_dir / "PER_QUERY.jsonl").open("w", encoding="utf-8") as handle:
                for tr in traces:
                    handle.write(json.dumps(tr, ensure_ascii=False) + "\n")
        cells[cell] = {
            "eval": ev,
            "train": train_stats,
            "rollout": gstat,
            "n_decision_points": gstat["n_decision_points"],
            "n_component_states": len(collected),
            "reward_unchanged_by_teacher": cell in {"teacher", "before"},
            "adapter": adapter_map.get(cell),
            "on_policy_refresh": refresh,
            "rollout_backend": "vllm" if vllm_on else "hf",
        }
        (cell_dir / "CELL.json").write_text(json.dumps(cells[cell], indent=2) + "\n", encoding="utf-8")
        eval_summaries.append(ev)
        if train_only:
            print(
                json.dumps(
                    {
                        "cell": cell,
                        "train_only": True,
                        "n_decision_points": gstat["n_decision_points"],
                        "n_optimizer_steps": (train_stats or {}).get("n_optimizer_steps"),
                        "adapter": adapter_map.get(cell),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "cell": cell,
                        "split": "official_test",
                        **{
                            k: ev.get(k)
                            for k in (
                                "n_queries",
                                "legal_action_rate",
                                "recall",
                                "trajectory_recall",
                                "final_answer_recall",
                                "precision",
                                "f1",
                                "reward",
                                "test_evidence_recall_at_5",
                                "mean_tool_calls_per_query",
                                "tool_search_cost",
                                "mean_e2e_sec",
                                "mean_model_sec",
                                "mean_harness_sec",
                            )
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        release_hf()

    write_reload_audit(out / "ADAPTER_RELOAD_AUDIT.json", adapter_audits)
    (out / "ADAPTER_MAP.json").write_text(json.dumps(adapter_map, indent=2) + "\n", encoding="utf-8")
    if train_only:
        official = {
            "skipped": True,
            "note": "train_only; score with scripts/run_eval.py",
        }
    else:
        official = write_eval_outputs(
            out,
            component_id=args.component,
            summaries=eval_summaries,
            adapter_audits=adapter_audits,
            pool_meta=pool_meta["eval"],
        )
    rl_opd = cells.get("rl_opd", {}).get("train") or {}
    scape_rl = cells.get("scape_rl", {}).get("train") or {}
    scape_seed = cells.get("scape_seed", {}).get("train") or {}
    joint = scape_seed or scape_rl or rl_opd
    joint_cell_present = "scape_seed" in cells or "scape_rl" in cells or "rl_opd" in cells
    summary = {
        "elapsed_sec": time.time() - t0,
        "manifest": manifest,
        "cells": {
            k: {kk: vv for kk, vv in v.items() if kk != "train"}
            | {"train": {tk: tv for tk, tv in (v.get("train") or {}).items() if tk != "call_log"}}
            for k, v in cells.items()
        },
        "official_eval": official,
        "q1_joint_one_optim": (
            int(joint.get("n_rl_forward_backward") or 0) >= 1
            and int(joint.get("n_opd_forward_backward") or 0) >= 1
            and int(joint.get("n_optimizer_steps") or 0) == (0 if not joint_cell_present else args.train_steps)
        ),
        "q2_on_policy_projection": any(c.get("n_decision_points") for c in cells.values()),
        "q3_teacher_does_not_change_reward": all(c.get("reward_unchanged_by_teacher") for c in cells.values()),
        "on_policy_refresh": bool(getattr(args, "on_policy_refresh", True)),
        "rollout_backend": "vllm" if vllm_on else "hf",
        "train_only": train_only,
    }
    summary_name = "TRAIN_SUMMARY.json" if train_only else "FOUR_CELL_SUMMARY.json"
    (out / summary_name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if train_only:
        (out / "FOUR_CELL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "RUN_COMPLETE").write_text(
        json.dumps({"ok": True, "elapsed_sec": summary.get("elapsed_sec"), "train_only": train_only}) + "\n",
        encoding="utf-8",
    )
    return summary


def coerce_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Accept both the four-cell CLI and run_true_scape_rl_opd flags."""
    if not hasattr(args, "component"):
        args.component = getattr(args, "target_component", "sentence_compress")
    if not hasattr(args, "train_steps"):
        args.train_steps = int(getattr(args, "max_steps", 64))
    if not hasattr(args, "n_queries") or getattr(args, "n_queries", None) in {None, 0}:
        if uses_sec_train_data(args):
            args.n_queries = None
        else:
            args.n_queries = BCPLUS_TRAIN
    if not hasattr(args, "sec_corpus_root") or getattr(args, "sec_corpus_root", None) in {None, ""}:
        args.sec_corpus_root = default_sec_corpus_root()
    if not hasattr(args, "rl_data") or getattr(args, "rl_data", None) in {None, ""}:
        args.rl_data = default_sec_rl_data()
    if not getattr(args, "score_split", None):
        args.score_split = SCORE_SPLIT_830 if uses_sec_train_data(args) else SCORE_SPLIT_166
    if not hasattr(args, "max_new_tokens"):
        args.max_new_tokens = 2048
    if not hasattr(args, "gpu"):
        args.gpu = "0"
    if not hasattr(args, "sft_adapter"):
        args.sft_adapter = getattr(args, "base_checkpoint", "") or ""
    if not hasattr(args, "base_model"):
        args.base_model = getattr(args, "base_checkpoint", "") or ""
    if not hasattr(args, "official_eval"):
        args.official_eval = True
    if not hasattr(args, "train_only"):
        args.train_only = False
    if not hasattr(args, "query_manifest"):
        args.query_manifest = None
    if not hasattr(args, "eval_manifest"):
        args.eval_manifest = None
    if not hasattr(args, "n_eval"):
        args.n_eval = None
    if not hasattr(args, "train_states"):
        args.train_states = None
    if not hasattr(args, "n_train_states"):
        args.n_train_states = None
    if not hasattr(args, "seeds"):
        args.seeds = [int(args.seed)]
    if not hasattr(args, "rollout_backend"):
        args.rollout_backend = "vllm"
    if not hasattr(args, "gpu_schedule"):
        args.gpu_schedule = "scheme_a"
    if not hasattr(args, "on_policy_refresh"):
        args.on_policy_refresh = True
    if not hasattr(args, "tensor_parallel_size"):
        args.tensor_parallel_size = None
    if not hasattr(args, "max_model_len"):
        args.max_model_len = 8192
    if getattr(args, "opd_states_per_trajectory", None) is None:
        args.opd_states_per_trajectory = (
            -1 if is_seed_scale_mode(args) else 3
        )
    if not getattr(args, "opd_loss", None):
        if getattr(args, "training_mode", "") == TRAINING_MODE_SCAPE_RL:
            args.opd_loss = OPD_LOSS_SAMPLED_GAP
        elif getattr(args, "training_mode", "") == TRAINING_MODE_SCAPE_SEED:
            args.opd_loss = OPD_LOSS_PROJECTED_GAP
        else:
            args.opd_loss = "sr_opd_ce"
    if getattr(args, "lambda_opd", None) is None:
        args.lambda_opd = (
            SCAPE_RL_LAMBDA_OPD
            if is_seed_scale_mode(args)
            else 0.1
        )
    if getattr(args, "opd_gate_beta", None) is None:
        args.opd_gate_beta = SCAPE_RL_OPD_GATE_BETA
    if not hasattr(args, "eval_max_turns"):
        from trim.eval.eval_defaults import HARNESS1_EVAL_MAX_TURNS

        args.eval_max_turns = HARNESS1_EVAL_MAX_TURNS
    if not hasattr(args, "eval_max_new_tokens"):
        from trim.eval.eval_defaults import HARNESS1_EVAL_MAX_NEW_TOKENS

        args.eval_max_new_tokens = HARNESS1_EVAL_MAX_NEW_TOKENS
    if not hasattr(args, "eval_temperature"):
        from trim.eval.eval_defaults import HARNESS1_EVAL_TEMPERATURE

        args.eval_temperature = HARNESS1_EVAL_TEMPERATURE
    if not hasattr(args, "train_device_map"):
        args.train_device_map = ""
    if not hasattr(args, "gpu_memory_utilization"):
        args.gpu_memory_utilization = 0.90
    if not hasattr(args, "enforce_eager"):
        args.enforce_eager = True
    if not hasattr(args, "vllm_python"):
        args.vllm_python = ""
    if not hasattr(args, "max_num_seqs"):
        args.max_num_seqs = 256
    if not hasattr(args, "vllm_generate_timeout_s"):
        args.vllm_generate_timeout_s = 3600.0
    if not hasattr(args, "vllm_disable_custom_all_reduce"):
        args.vllm_disable_custom_all_reduce = None
    if not hasattr(args, "train_groups_per_step"):
        args.train_groups_per_step = HF_DEFAULT_GROUPS_PER_STEP
    if not hasattr(args, "train_micro_batch_size"):
        args.train_micro_batch_size = HF_DEFAULT_MICRO_BATCH
    if not hasattr(args, "train_heartbeat_every"):
        args.train_heartbeat_every = HF_DEFAULT_HEARTBEAT_EVERY
    if getattr(args, "smoke", False):
        if getattr(args, "n_queries", None) in {None, 0}:
            args.n_queries = 6
        else:
            args.n_queries = min(int(args.n_queries), 6)
        args.group_size = min(int(args.group_size), 2)
        args.max_turns = min(int(args.max_turns), 2)
        args.train_steps = min(int(args.train_steps), 1)
        args.max_new_tokens = min(int(args.max_new_tokens), 256)
        args.eval_max_turns = min(int(getattr(args, "eval_max_turns", 2)), 2)
        args.eval_max_new_tokens = min(int(getattr(args, "eval_max_new_tokens", 256)), 256)
        args.n_eval = 6 if args.n_eval is None else min(int(args.n_eval), 6)
    return args


def run_seeded_four_cell(args: argparse.Namespace) -> dict[str, Any]:
    """Train seed 42/43 (or --seeds) from the same launch, each with its own adapter + manifest."""
    args = coerce_runtime_args(args)
    seeds = [int(x) for x in (getattr(args, "seeds", None) or [args.seed])]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in seeds:
        child = argparse.Namespace(**vars(args))
        child.seed = seed
        child.out = root / f"seed{seed}"
        tag = "train" if bool(getattr(args, "train_only", False)) else "four_cell"
        print(f"[{tag}] seed={seed} out={child.out}", flush=True)
        per_seed[str(seed)] = run_four_cell(child)
    payload = {
        "component": args.component,
        "opd_loss": str(getattr(args, "opd_loss", None) or "sr_opd_ce"),
        "rl_loss_fn": "cispo",
        "seeds": seeds,
        "score_split": SCORE_SPLIT_830 if uses_bcplus_830_eval(args) else SCORE_SPLIT_166,
        "legacy_adapters_not_used": True,
        "per_seed": {k: {"q1": v.get("q1_joint_one_optim"), "q2": v.get("q2_on_policy_projection"), "q3": v.get("q3_teacher_does_not_change_reward"), "out": str(root / f"seed{k}")} for k, v in per_seed.items()},
    }
    (root / "SEEDED_FOUR_CELL_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_from_rl_opd_args(args: argparse.Namespace) -> dict[str, Any]:
    """Live path for run_true_scape_rl_opd.py."""
    args = coerce_runtime_args(args)
    if getattr(args, "validate_only", False) or getattr(args, "dry_run", False):
        report = validate_wiring(args)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "VALIDATE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    if len(getattr(args, "seeds", [args.seed]) or [args.seed]) > 1:
        return run_seeded_four_cell(args)
    return run_four_cell(args)
