"""Formal Before / RL / PURE / RL+OPD loop for sr_opd_ce + CISPO.

backend=hf_debug. Each hybrid substep is CISPO FB → CE FB → one optim_step.
Teacher is a side branch and must not change RL rewards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from scape.adapters.components import minus_mask
from scape.eval.adapter_reload_audit import audit_saved_adapter, write_reload_audit
from scape.eval.browsecomp_retrieval import RetrievalBackend, hits_to_doc_store, open_retrieval
from scape.eval.official_query_pool import attach_bcp_fields, load_official_384, load_train_queries, overlap_ids
from scape.eval.sr_opd_four_cell_eval import search_metrics, summarize_traces, write_eval_outputs
from scape.state.snapshot import capture_snapshot
from scape.training.action_codec import STUDENT_NATIVE_TOOLS, render_action
from scape.training.hf_rl_opd_client import (
    HFDebugTrainingClient,
    group_relative_advantages,
    restore_trainable,
    snapshot_trainable,
)
from scape.training.on_policy_collector import filter_component_states, write_collected_states
from scape.training.opd_dataset import render_student_prompt
from scape.training.rl_opd_types import (
    PROTOCOL_COMPLETE_RL_OPD,
    TRAINING_MODE_PURE_OPD,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
    HybridRolloutGroup,
    StudentDecisionPoint,
)
from scape.training.sentence_compress_teacher import teacher_events_from_point
from scape.training.tinker_rl_opd_trainer import hybrid_train_substep, prepare_hybrid_batch

CELLS = ("before", "rl", "pure_opd", "rl_opd")
TeacherFn = Callable[[StudentDecisionPoint], list[Any]]

TEACHER_REGISTRY: dict[str, TeacherFn] = {
    "sentence_compress": teacher_events_from_point,
}


def teacher_for(component_id: str) -> TeacherFn | None:
    return TEACHER_REGISTRY.get(component_id)


def cell_lambda(name: str, lambda_opd: float) -> float:
    if name in {"before", "rl"}:
        return 0.0
    return float(lambda_opd)


def cells_for_mode(training_mode: str | None) -> tuple[str, ...]:
    if training_mode in {None, "", "four_cell"}:
        return CELLS
    if training_mode == TRAINING_MODE_RL:
        return ("before", "rl")
    if training_mode == TRAINING_MODE_PURE_OPD:
        return ("before", "pure_opd")
    if training_mode == TRAINING_MODE_RL_OPD:
        return ("before", "rl_opd")
    return CELLS


def build_manifest(args: argparse.Namespace, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = getattr(args, "training_mode", "four_cell")
    lam = 0.0 if mode == TRAINING_MODE_RL else float(args.lambda_opd)
    return {
        "training_mode": mode,
        "component": args.component,
        "target_component": args.component,
        "rl_loss_fn": "cispo",
        "opd_loss": "sr_opd_ce",
        "lambda_opd": lam,
        "student_harness": "H_min",
        "teacher_harness": "H_full",
        "opd_state_source": "current_on_policy_rl_rollout",
        "joint_update_contract": "rl_fb+opd_fb+single_optim",
        "legacy_tool_token_kl_hook_used": False,
        "protocol_complete_rl_opd": mode in {"four_cell", TRAINING_MODE_RL_OPD} and lam > 0,
        "protocol_name": PROTOCOL_COMPLETE_RL_OPD,
        "projection_schema_version": "scape_projection_v1",
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "train_steps": args.train_steps,
        "n_queries": args.n_queries,
        "opd_states_per_trajectory": args.opd_states_per_trajectory,
        "seed": args.seed,
        "base_model": args.base_model,
        "sft_adapter": getattr(args, "sft_adapter", ""),
        "scale": "smoke" if getattr(args, "smoke", False) else "full",
        "backend": "hf_debug",
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


def doc_store_for_row(row: dict[str, Any], searcher: RetrievalBackend | None) -> dict[str, Any]:
    if searcher is not None and searcher.name != "none":
        hits = searcher.search(str(row.get("query") or ""), 12)
        store = hits_to_doc_store(hits)
        if store:
            return store
    return labeled_doc_store(row)


def snap_from_state(qid: str, st: dict[str, Any], component_id: str):
    curated = [str(x) for x in (st.get("curated") or {})]
    pool = [str(x) for x in (st.get("pool") or {})]
    store = st.get("doc_store") or {}
    documents = []
    for did, rec in list(store.items())[:16]:
        if isinstance(rec, dict):
            documents.append({"id": str(did), "text": str(rec.get("text") or "")[:2000]})
        else:
            documents.append({"id": str(did), "text": str(rec)[:2000]})
    return capture_snapshot(
        query_id=qid,
        step=int(st.get("step") or 0),
        harness_mask=minus_mask(component_id),
        working_memory={
            "curated_ids": curated,
            "accessible_doc_ids": list(dict.fromkeys(pool + curated + list(store))),
            "pool": st.get("pool") or {},
            "documents": documents,
            "query": st.get("query"),
            "doc_store": {did: {"id": did, "text": str((rec or {}).get("text") if isinstance(rec, dict) else rec)[:800]} for did, rec in list(store.items())[:12]},
        },
        tool_history=list(st.get("tool_history") or []),
        observations=[],
        metadata={"component_id": component_id, "owner": "student_reduced"},
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
    from scape.eval.local_search_env import curated_recall

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


def parse_generated_action(text: str, completion_ids: list[int] | None, enc) -> tuple[dict[str, Any], bool]:
    from scape.eval.harmony_runtime import parse_harmony_tool_call

    parsed = parse_harmony_tool_call(text, completion_ids=completion_ids, enc=enc)
    name = parsed.tool_name
    if parsed.legal and name in STUDENT_NATIVE_TOOLS:
        return {"name": name, "arguments": dict(parsed.arguments or {})}, True
    return {"name": name or "unknown", "arguments": dict(parsed.arguments or {})}, False


def _maybe_empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_harmony(backend, query: str, *, enc, max_new: int, sample: bool, seed: int, prompt_ids: list[int] | None = None) -> dict[str, Any]:
    import torch
    from scape.eval.harmony_runtime import (
        build_first_turn_prompt_ids,
        decode_ids,
        stop_ids_for_tool_actions,
    )

    ids = list(prompt_ids or build_first_turn_prompt_ids(query, enc=enc))
    stop_ids = stop_ids_for_tool_actions(enc)
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
) -> tuple[list[StudentDecisionPoint], list[dict[str, Any]], float, dict[str, Any]]:
    from scape.eval.harmony_runtime import (
        build_continuation_prompt_ids,
        build_first_turn_prompt_ids,
        make_action,
        make_observation,
    )
    from scape.eval.local_search_env import curated_recall, execute_tool, new_state, wm_text
    import torch

    query = str(row["query"])
    qid = str(row["query_id"])
    gold_ids = [str(x) for x in (row.get("gold_docids") or row.get("evidence_docids") or [])]
    st = new_state(query, doc_store_for_row(row, searcher))
    acts: list[tuple[Any, Any]] = []
    points: list[StudentDecisionPoint] = []
    rows: list[dict[str, Any]] = []
    valids: list[bool] = []
    actions: list[dict[str, Any]] = []
    names: list[str] = []
    for turn in range(max_turns):
        if st.get("ended"):
            break
        if turn == 0:
            pids = build_first_turn_prompt_ids(query, enc=enc)
        else:
            pids = build_continuation_prompt_ids(query, actions_obs=acts, wm_text=wm_text(st, auto_on=False), enc=enc)
        pre = snap_from_state(qid, st, component_id)
        student_prefix = render_student_prompt(pre, component_id=component_id)
        gen = generate_harmony(
            backend,
            query,
            enc=enc,
            max_new=max_new,
            sample=sample,
            seed=seed + 17 * rollout_idx + turn,
            prompt_ids=pids,
        )
        action, valid = parse_generated_action(gen["text"], gen["action_ids"], enc)
        valids.append(valid)
        actions.append(action)
        names.append(str(action.get("name")))
        st, obs, _ok = execute_tool(st, action.get("name") if valid else None, action.get("arguments"))
        if valid:
            try:
                acts.append((make_action(action["name"], action.get("arguments") or {}), make_observation(obs)))
            except Exception:
                pass
        action_ids = list(gen["action_ids"]) or backend.encode(render_action(action) if valid else "to=unknown\n{}\n")
        prompt_ids = list(gen["prompt_ids"])
        with torch.no_grad():
            old_prompt = prompt_ids[-384:] if len(prompt_ids) > 384 else prompt_ids
            old_act = action_ids[:128]
            old_lp = backend._teacher_forced_logprobs(old_prompt, old_act, require_grad=False)
        old_mean = float(old_lp.mean().item()) if old_lp.numel() else 0.0
        post = snap_from_state(qid, st, component_id)
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
            )
        )
        rows.append(
            {
                "query_id": qid,
                "prompt": gen.get("prompt_text") or student_prefix,
                "prompt_ids": prompt_ids,
                "action_text": gen["text"],
                "action_ids": action_ids,
                "logprob_old": old_mean,
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
    stats = {
        "names": names,
        "reward": reward,
        "n_curated": len(st.get("curated") or {}),
        "gold_recall": float(curated_recall(st, gold_ids) or 0.0),
        "ended": bool(st.get("ended")),
        "n_turns": len(names),
        "n_tool_calls": int(st.get("n_tool_calls") or 0),
        "n_search_calls": int(st.get("n_search_calls") or 0),
        "search_query": next((a.get("arguments", {}).get("query") for a in actions if a.get("name") == "search_corpus"), query),
    }
    return points, rows, reward, stats


def rollout_group(backend, *, row, component_id, group_size, max_turns, max_new, policy_version, seed, sample, enc, searcher=None) -> HybridRolloutGroup:
    points: list[StudentDecisionPoint] = []
    rewards: list[float] = []
    rl_rows: list[dict[str, Any]] = []
    tool_seqs: list[list[str]] = []
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
        )
        points.extend(ep_points)
        rl_rows.extend(ep_rows)
        rewards.append(reward)
        tool_seqs.append(list(stats["names"]))
    adv = group_relative_advantages([r["reward"] for r in rl_rows], [r["query_id"] for r in rl_rows])
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
) -> dict[str, Any]:
    if name == "before" or train_steps <= 0:
        return {
            "update_type": "eval_only",
            "n_optimizer_steps": 0,
            "n_rl_forward_backward": 0,
            "n_opd_forward_backward": 0,
            "skipped_teacher": True,
        }
    client = HFDebugTrainingClient(backend)
    rl_by_q = {g.query_id: list((g.trajectory_group or {}).get("rl_rows") or []) for g in groups}
    teacher = None if lambda_opd <= 0 else teacher_fn
    metrics_acc: list[dict[str, Any]] = []
    last_batch_stats: dict[str, Any] = {}
    for step in range(train_steps):
        batch = prepare_hybrid_batch(
            groups=groups,
            rl_datums_by_query=rl_by_q,
            policy_version=policy_version,
            lambda_opd=lambda_opd,
            component_id=component_id,
            teacher_event_fn=teacher,
            encode_fn=backend.encode,
            opd_states_per_trajectory=opd_states_per_trajectory,
            seed=step,
            remove_constant_reward_groups=True,
        )
        last_batch_stats = batch.projection_stats
        if name == "pure_opd":
            rl_use, opd_use = [], batch.opd_datums
        elif name == "rl":
            rl_use, opd_use = batch.rl_datums, []
        else:
            rl_use, opd_use = batch.rl_datums, batch.opd_datums
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
        )
        metrics_acc.append(m.to_dict())
    return {
        "call_log": list(client.calls),
        "n_optimizer_steps": sum(1 for c in client.calls if c[0] == "opt"),
        "n_rl_forward_backward": sum(1 for c in client.calls if c[:2] == ("fb", "cispo")),
        "n_opd_forward_backward": sum(1 for c in client.calls if c[:2] == ("fb", "cross_entropy")),
        "projection_stats": last_batch_stats,
        "substeps": metrics_acc,
        "backend": HFDebugTrainingClient.backend_name,
    }


def eval_closed_loop(backend, rows: list[dict[str, Any]], *, component_id, max_new, max_turns, seed, enc, searcher) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
            sample=False,
            enc=enc,
            rollout_idx=0,
            searcher=searcher,
        )
        prefix = render_student_prompt(snap_from_state(row["query_id"], {"query": row["query"], "doc_store": {}, "curated": {}, "pool": {}}, component_id), component_id=component_id)
        if "compressed_teacher_view" in prefix or "VERIFY_RESULT_SECRET" in prefix:
            leak += 1
        search_q = str(stats.get("search_query") or row["query"])
        sm = search_metrics(searcher, search_q, list(row.get("evidence_docids") or [])) if searcher is not None else {}
        traces.append(
            {
                "query_id": row["query_id"],
                "tool_names": list(stats["names"]),
                "reward": reward,
                "gold_recall": stats["gold_recall"],
                "n_tool_calls": stats["n_tool_calls"],
                "n_search_calls": stats["n_search_calls"],
                **sm,
            }
        )
    retrieval_name = searcher.name if searcher is not None else "none"
    summary = summarize_traces(traces, setting="closed_loop", retrieval_name=retrieval_name)
    summary["teacher_leak_rate"] = leak / max(1, len(rows))
    summary["mean_reward"] = sum(t["reward"] for t in traces) / max(1, len(traces))
    summary["mean_gold_recall"] = sum(float(t["gold_recall"]) for t in traces) / max(1, len(traces))
    return summary, traces


def resolve_queries(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eval_rows, eval_meta = load_official_384(manifest=getattr(args, "eval_manifest", None))
    train_path = getattr(args, "query_manifest", None)
    train_rows, train_meta = load_train_queries(
        manifest=Path(train_path) if train_path else None,
        n_queries=args.n_queries,
        exclude_eval_ids={r["query_id"] for r in eval_rows},
    )
    train_rows = attach_bcp_fields(train_rows)
    overlap = overlap_ids(train_rows, eval_rows)
    if overlap:
        raise RuntimeError(f"train/eval query overlap: {overlap[:8]}")
    return train_rows, eval_rows, {"train": train_meta, "eval": eval_meta, "overlap": overlap}


def validate_wiring(args: argparse.Namespace) -> dict[str, Any]:
    from scape.state.snapshot import EnvironmentSnapshot
    from scape.training.opd_dataset import project_and_materialize
    from scape.training.opd_projection import StudentActionSpaceProjector
    from scape.training.sentence_compress_teacher import teacher_events_from_wm

    teacher_fn = teacher_for(args.component)
    if teacher_fn is None:
        raise SystemExit(f"no teacher registered for component={args.component}")
    train_rows, eval_rows, pool_meta = resolve_queries(args)
    wm = {
        "query": train_rows[0]["query"],
        "documents": [{"id": "d_long", "text": ("Long noisy passage. " * 40) + train_rows[0]["query"]}],
        "curated_ids": [],
    }
    events = teacher_events_from_wm(wm)
    snap = capture_snapshot(
        query_id=train_rows[0]["query_id"],
        step=0,
        harness_mask=minus_mask(args.component),
        working_memory=wm,
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
        "teacher_registered": True,
        "n_train_queries": len(train_rows),
        "n_eval_queries": len(eval_rows),
        "eval_is_official_384": int(pool_meta["eval"]["query_count"]) == 384,
        "projection_kind": projection.kind.value,
        "n_projected_steps": len(steps),
        "teacher_leak_in_student_prefix": leaked,
        "pool": pool_meta,
        "snapshot_type": type(snap).__name__,
        "environment_snapshot": EnvironmentSnapshot.__name__,
    }


def run_four_cell(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from scape.eval.harmony_runtime import load_harmony_enc
    from scape.training.hf_tool_opd import ScapeHFToolOPD

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows, pool_meta = resolve_queries(args)
    searcher = open_retrieval()
    manifest = build_manifest(
        args,
        extra={"pool": pool_meta, "retrieval": searcher.name, "cells": list(cells_for_mode(getattr(args, "training_mode", "four_cell")))},
    )
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    teacher_fn = teacher_for(args.component)
    if teacher_fn is None:
        raise SystemExit(f"no teacher registered for {args.component}")

    model_src = args.sft_adapter if args.sft_adapter and Path(args.sft_adapter).exists() else args.base_model
    t0 = time.time()
    backend = ScapeHFToolOPD(
        model_path=model_src,
        device_map=f"cuda:{int(args.gpu)}",
        learning_rate=1e-5,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    )
    theta0 = snapshot_trainable(backend.model)
    enc = load_harmony_enc()
    chosen_cells = cells_for_mode(getattr(args, "training_mode", "four_cell"))
    adapter_map: dict[str, str | None] = {}
    adapter_audits: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    eval_summaries: list[dict[str, Any]] = []

    for cell in chosen_cells:
        restore_trainable(backend.model, theta0)
        backend.optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad], lr=1e-5)
        print(f"[four_cell] cell={cell} rollout", flush=True)
        groups = [
            rollout_group(
                backend,
                row=row,
                component_id=args.component,
                group_size=args.group_size,
                max_turns=args.max_turns,
                max_new=args.max_new_tokens,
                policy_version="v0",
                seed=args.seed + 100 * (abs(hash(cell)) % 1000),
                sample=True,
                enc=enc,
                searcher=searcher,
            )
            for row in train_rows
        ]
        collected = filter_component_states(
            [p for g in groups for p in g.decision_points],
            component_id=args.component,
            require_valid=False,
        )
        write_collected_states(
            collected,
            out / cell / "collected_states.jsonl",
            component_id=args.component,
            extra={"cell": cell, "n_rollout_points": sum(len(g.decision_points) for g in groups)},
        )
        gstat = group_stats(groups)
        rewards_before = [r for g in groups for r in g.terminal_rewards]
        train_stats = asyncio.run(
            train_cell(
                name=cell,
                backend=backend,
                groups=groups,
                lambda_opd=cell_lambda(cell, args.lambda_opd),
                train_steps=0 if cell == "before" else args.train_steps,
                policy_version="v0",
                opd_states_per_trajectory=args.opd_states_per_trajectory,
                component_id=args.component,
                teacher_fn=teacher_fn,
            )
        )
        rewards_after = [r for g in groups for r in g.terminal_rewards]
        if rewards_before != rewards_after:
            raise RuntimeError("Teacher shadow mutated RL rewards")
        adapter_dir = out / "adapters" / cell
        restore_trainable(backend.model, theta0)
        reload_path = "theta0_no_adapter"
        if cell != "before":
            adapter_dir.mkdir(parents=True, exist_ok=True)
            backend.save_pretrained(str(adapter_dir))
            adapter_map[cell] = str(adapter_dir)
            file_audit = audit_saved_adapter(adapter_dir, cell=cell)
            from safetensors.torch import load_file
            from scape.eval.adapter_reload_audit import remap_lora_state

            weights = remap_lora_state(load_file(str(adapter_dir / "adapter_model.safetensors")))
            missing, unexpected = backend.model.load_state_dict(weights, strict=False)
            lora_missing = [x for x in missing if "lora_" in x]
            if lora_missing:
                raise RuntimeError(f"adapter reload failed for {cell}: {lora_missing}")
            reload_path = "saved_adapter_state_dict"
            file_audit["reload_path"] = reload_path
            file_audit["unexpected_lora"] = [x for x in unexpected if "lora_" in x]
            adapter_audits.append(file_audit)
        else:
            adapter_map[cell] = None
            adapter_audits.append({"cell": cell, "adapter_dir": None, "reload_ready": True, "exists": False, "reload_path": reload_path})
        ev_rows = eval_rows if getattr(args, "official_eval", True) else train_rows
        if getattr(args, "n_eval", None):
            ev_rows = ev_rows[: int(args.n_eval)]
        ev, traces = eval_closed_loop(
            backend,
            ev_rows,
            component_id=args.component,
            max_new=args.max_new_tokens,
            max_turns=args.max_turns,
            seed=args.seed,
            enc=enc,
            searcher=searcher,
        )
        ev["setting"] = cell
        cell_dir = out / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        with (cell_dir / "PER_QUERY.jsonl").open("w", encoding="utf-8") as handle:
            for tr in traces:
                handle.write(json.dumps(tr, ensure_ascii=False) + "\n")
        cells[cell] = {
            "eval": ev,
            "train": train_stats,
            "rollout": gstat,
            "n_decision_points": gstat["n_decision_points"],
            "n_component_states": len(collected),
            "reward_unchanged_by_teacher": True,
            "adapter": adapter_map[cell],
        }
        (cell_dir / "CELL.json").write_text(json.dumps(cells[cell], indent=2) + "\n", encoding="utf-8")
        eval_summaries.append(ev)
        print(json.dumps({"cell": cell, **{k: ev.get(k) for k in ("legal_action_rate", "test_evidence_recall_at_5", "mean_tool_calls_per_query", "tool_search_cost")}}, ensure_ascii=False), flush=True)

    write_reload_audit(out / "ADAPTER_RELOAD_AUDIT.json", adapter_audits)
    (out / "ADAPTER_MAP.json").write_text(json.dumps(adapter_map, indent=2) + "\n", encoding="utf-8")
    official = write_eval_outputs(
        out,
        component_id=args.component,
        summaries=eval_summaries,
        adapter_audits=adapter_audits,
        pool_meta=pool_meta["eval"],
    )
    rl_opd = cells.get("rl_opd", {}).get("train") or {}
    summary = {
        "elapsed_sec": time.time() - t0,
        "manifest": manifest,
        "cells": {k: {kk: vv for kk, vv in v.items() if kk != "train"} | {"train": {tk: tv for tk, tv in (v.get("train") or {}).items() if tk != "call_log"}} for k, v in cells.items()},
        "official_eval": official,
        "q1_joint_one_optim": (
            int(rl_opd.get("n_rl_forward_backward") or 0) >= 1
            and int(rl_opd.get("n_opd_forward_backward") or 0) >= 1
            and int(rl_opd.get("n_optimizer_steps") or 0) == (0 if "rl_opd" not in cells else args.train_steps)
        ),
        "q2_on_policy_projection": any(c.get("n_decision_points") for c in cells.values()),
        "q3_teacher_does_not_change_reward": all(c.get("reward_unchanged_by_teacher") for c in cells.values()),
    }
    (out / "FOUR_CELL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def coerce_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Accept both the four-cell CLI and run_true_scape_rl_opd flags."""
    if not hasattr(args, "component"):
        args.component = getattr(args, "target_component", "sentence_compress")
    if not hasattr(args, "train_steps"):
        args.train_steps = int(getattr(args, "max_steps", 64))
    if not hasattr(args, "n_queries"):
        args.n_queries = int(getattr(args, "batch_size", 32))
    if not hasattr(args, "max_new_tokens"):
        args.max_new_tokens = 384
    if not hasattr(args, "gpu"):
        args.gpu = 0
    if not hasattr(args, "sft_adapter"):
        args.sft_adapter = getattr(args, "base_checkpoint", "") or ""
    if not hasattr(args, "base_model"):
        args.base_model = getattr(args, "base_checkpoint", "") or ""
    if not hasattr(args, "official_eval"):
        args.official_eval = True
    if not hasattr(args, "query_manifest"):
        args.query_manifest = None
    if not hasattr(args, "eval_manifest"):
        args.eval_manifest = None
    if not hasattr(args, "n_eval"):
        args.n_eval = None
    if getattr(args, "smoke", False):
        args.n_queries = min(int(args.n_queries), 6)
        args.group_size = min(int(args.group_size), 2)
        args.max_turns = min(int(args.max_turns), 2)
        args.train_steps = min(int(args.train_steps), 1)
        args.max_new_tokens = min(int(args.max_new_tokens), 256)
        args.n_eval = 6 if args.n_eval is None else args.n_eval
    return args


def run_from_rl_opd_args(args: argparse.Namespace) -> dict[str, Any]:
    """Live path for run_true_scape_rl_opd.py."""
    args = coerce_runtime_args(args)
    if getattr(args, "validate_only", False) or getattr(args, "dry_run", False):
        report = validate_wiring(args)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "VALIDATE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    return run_four_cell(args)
