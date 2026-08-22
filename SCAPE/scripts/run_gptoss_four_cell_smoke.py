#!/usr/bin/env python3
"""gpt-oss-20b 4-cell runner: Before / RL / PURE_SR_OPD / RL+SR-OPD.

Defaults are the formal H_min 四格 scale (64 queries, group 8, 6 turns, 8
optimizer steps). Pass --smoke for the 6×2×2×1 debug config.

backend=hf_debug. CISPO is not rewritten; grads accumulate then one optim_step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

_SCAPE = Path(__file__).resolve().parents[1]
_SCOPE = _SCAPE.parent / "SCOPE"
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))
if str(_SCOPE) not in sys.path:
    sys.path.insert(0, str(_SCOPE))

from scape.adapters.components import minus_mask
from scape.eval.harmony_runtime import (
    build_continuation_prompt_ids,
    build_first_turn_prompt_ids,
    decode_ids,
    load_harmony_enc,
    make_action,
    make_observation,
    parse_harmony_tool_call,
    stop_ids_for_tool_actions,
)
from scape.eval.local_search_env import curated_recall, execute_tool, new_state, wm_text
from scape.state.snapshot import capture_snapshot
from scape.training.action_codec import STUDENT_NATIVE_TOOLS, render_action
from scape.training.hf_rl_opd_client import (
    HFDebugTrainingClient,
    group_relative_advantages,
    restore_trainable,
    snapshot_trainable,
)
from scape.training.hf_tool_opd import ScapeHFToolOPD
from scape.training.opd_dataset import render_student_prompt
from scape.training.opd_events import harness_mutation
from scape.training.rl_opd_types import HybridRolloutGroup, StudentDecisionPoint
from scape.training.tinker_rl_opd_trainer import hybrid_train_substep, prepare_hybrid_batch

COMPONENT = "auto_populate_first_search"
CELLS = ("before", "rl", "pure_opd", "rl_opd")
# Formal closed-loop 四格 used n=64; these seeds are expanded if n_queries is larger.
QUERY_SEEDS = [
    "What is the filing date of the 10-K?",
    "Who is the CEO mentioned in the report?",
    "Which exhibit lists the risk factors?",
    "When was the subsidiary acquired?",
    "What revenue did the company report?",
    "Where is the headquarters located?",
    "Which auditor signed the opinion?",
    "What is the ticker symbol?",
    "What was the year-over-year change in operating income?",
    "How many shares were outstanding at year end?",
    "What dividend was declared in the latest quarter?",
    "Which segment contributed the most revenue?",
    "What is the company's effective tax rate?",
    "When does the revolving credit facility mature?",
    "What material weakness did the auditor identify?",
    "Which related-party transaction is disclosed?",
]


def expand_queries(n: int) -> list[str]:
    out = list(QUERY_SEEDS)
    extra = 1
    while len(out) < n:
        for seed in QUERY_SEEDS:
            if len(out) >= n:
                break
            out.append(f"{seed} Additional detail request {extra}.")
        extra += 1
    return out[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gpt-oss-20b 4-cell RL / PURE / RL+OPD")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    p.add_argument(
        "--sft-adapter",
        default=str(
            _SCAPE / "outputs/0814_clean_mechanism/sft/gpu0/full_s42_full/lora_checkpoint"
        ),
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-queries", type=int, default=64)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--lambda-opd", type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--opd-states-per-trajectory", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Override defaults to 6 queries / group 2 / 2 turns / 1 step.",
    )
    return p.parse_args()


def apply_scale(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.n_queries = 6
        args.group_size = 2
        args.max_turns = 2
        args.train_steps = 1
        args.max_new_tokens = 256
        args.opd_states_per_trajectory = 2
    return args


def parse_generated_action(text: str, completion_ids: list[int] | None = None, enc=None) -> tuple[dict[str, Any], bool]:
    parsed = parse_harmony_tool_call(text, completion_ids=completion_ids, enc=enc)
    name = parsed.tool_name
    if parsed.legal and name in STUDENT_NATIVE_TOOLS:
        return {"name": name, "arguments": dict(parsed.arguments or {})}, True
    return {"name": name or "unknown", "arguments": dict(parsed.arguments or {})}, False


def doc_store_for(query: str) -> dict[str, Any]:
    return {
        "d1": {"id": "d1", "text": "alpha background notes about exhibits and weather."},
        "d2": {
            "id": "d2",
            "text": (
                f"beta gold evidence for the question: {query} "
                "The CEO signed the FY2023 10-K filed on 2023-11-03. "
                "Risk factors appear in Exhibit 99. The subsidiary was acquired in 2019. "
                "Revenue was 12.4 billion. Headquarters is in Cupertino. "
                "The auditor is PricewaterhouseCoopers. Ticker is AAPL."
            ),
        },
        "d3": {"id": "d3", "text": "noise document about sports scores and travel delays."},
    }


def snap_from_state(qid: str, st: dict[str, Any]):
    curated = [str(x) for x in (st.get("curated") or {})]
    pool = [str(x) for x in (st.get("pool") or {})]
    store = [str(x) for x in (st.get("doc_store") or {})]
    return capture_snapshot(
        query_id=qid,
        step=int(st.get("step") or 0),
        harness_mask=minus_mask(COMPONENT),
        working_memory={
            "curated_ids": curated,
            "accessible_doc_ids": list(dict.fromkeys(pool + curated + store)),
            "pool": pool,
            "documents": [{"id": did} for did in store],
            "query": st.get("query"),
        },
        tool_history=list(st.get("tool_history") or []),
        observations=[],
        metadata={"component_id": COMPONENT, "owner": "student_reduced"},
    )


def _query_overlap(action: dict[str, Any], query: str) -> float:
    args = action.get("arguments") or {}
    blob = " ".join(
        [str(args.get("query") or "")] + [str(x) for x in (args.get("queries") or [])]
    ).lower()
    qset = set(re.findall(r"[a-z0-9]+", query.lower()))
    aset = set(re.findall(r"[a-z0-9]+", blob))
    if not qset or not aset:
        return 0.0
    return 0.1 * len(qset & aset) / len(qset)


def terminal_reward(st: dict[str, Any], *, query: str, valids: list[bool], actions: list[dict[str, Any]]) -> float:
    if not valids or not any(valids):
        return -0.2
    rec = float(curated_recall(st, ["d2"]) or 0.0)
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


@torch.no_grad()
def generate_harmony(
    backend: ScapeHFToolOPD,
    query: str,
    *,
    enc,
    max_new: int,
    sample: bool,
    seed: int,
    prompt_ids: list[int] | None = None,
) -> dict[str, Any]:
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
        kw = dict(
            attention_mask=attn,
            max_new_tokens=max_new,
            eos_token_id=stop_ids,
            pad_token_id=stop_ids[0],
        )
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


def _maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def teacher_from_point(point: StudentDecisionPoint):
    wm = point.pre_action_snapshot.working_memory
    curated = [str(x) for x in (wm.get("curated_ids") or [])]
    accessible = [str(x) for x in (wm.get("accessible_doc_ids") or wm.get("pool") or [])]
    add = [did for did in accessible if did not in curated]
    if "d2" in accessible and "d2" not in curated:
        add = ["d2"] + [x for x in add if x != "d2"]
    after = list(dict.fromkeys(curated + add[:2]))
    if after == curated:
        after = list(dict.fromkeys(curated + ["d2"]))
    return [
        harness_mutation(
            COMPONENT,
            {"before_curated": curated, "after_curated": after},
        )
    ]


def _one_episode(
    backend: ScapeHFToolOPD,
    *,
    query: str,
    query_id: str,
    max_turns: int,
    max_new: int,
    policy_version: str,
    seed: int,
    sample: bool,
    enc,
    rollout_idx: int,
) -> tuple[list[StudentDecisionPoint], list[dict[str, Any]], float, dict[str, Any]]:
    st = new_state(query, doc_store_for(query))
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
            pids = build_continuation_prompt_ids(
                query,
                actions_obs=acts,
                wm_text=wm_text(st, auto_on=False),
                enc=enc,
            )
        pre = snap_from_state(query_id, st)
        student_prefix = render_student_prompt(pre, component_id=COMPONENT)
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
        action_ids = list(gen["action_ids"]) or backend.encode(
            render_action(action) if valid else "to=unknown\n{}\n"
        )
        prompt_ids = list(gen["prompt_ids"])
        with torch.no_grad():
            old_prompt = prompt_ids[-384:] if len(prompt_ids) > 384 else prompt_ids
            old_act = action_ids[:128]
            old_lp = backend._teacher_forced_logprobs(old_prompt, old_act, require_grad=False)
        old_mean = float(old_lp.mean().item()) if old_lp.numel() else 0.0
        post = snap_from_state(query_id, st)
        points.append(
            StudentDecisionPoint(
                episode_id=f"{query_id}_r{rollout_idx}",
                query_id=query_id,
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
                "query_id": query_id,
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
    reward = terminal_reward(st, query=query, valids=valids, actions=actions)
    for point in points:
        point.reward = reward
    for row in rows:
        row["reward"] = reward
    stats = {
        "names": names,
        "reward": reward,
        "n_curated": len(st.get("curated") or {}),
        "gold_recall": float(curated_recall(st, ["d2"]) or 0.0),
        "ended": bool(st.get("ended")),
        "n_turns": len(names),
    }
    return points, rows, reward, stats


def rollout_group(
    backend: ScapeHFToolOPD,
    *,
    query: str,
    query_id: str,
    group_size: int,
    max_turns: int,
    max_new: int,
    policy_version: str,
    seed: int,
    sample: bool,
    enc,
) -> HybridRolloutGroup:
    points: list[StudentDecisionPoint] = []
    rewards: list[float] = []
    rl_rows: list[dict[str, Any]] = []
    tool_seqs: list[list[str]] = []
    for g in range(group_size):
        ep_points, ep_rows, reward, stats = _one_episode(
            backend,
            query=query,
            query_id=query_id,
            max_turns=max_turns,
            max_new=max_new,
            policy_version=policy_version,
            seed=seed,
            sample=sample,
            enc=enc,
            rollout_idx=g,
        )
        points.extend(ep_points)
        rl_rows.extend(ep_rows)
        rewards.append(reward)
        tool_seqs.append(list(stats["names"]))
    adv = group_relative_advantages(
        [row["reward"] for row in rl_rows],
        [row["query_id"] for row in rl_rows],
    )
    for row, a in zip(rl_rows, adv):
        row["advantage"] = a
    return HybridRolloutGroup(
        query_id=query_id,
        policy_version=policy_version,
        trajectory_group={"rl_rows": rl_rows, "query": query, "tool_seqs": tool_seqs},
        decision_points=points,
        terminal_rewards=rewards,
        metadata={"n_rl_rows": len(rl_rows), "reward_spread": max(rewards) - min(rewards) if rewards else 0.0},
    )


def eval_cell(
    backend: ScapeHFToolOPD,
    queries: list[tuple[str, str]],
    *,
    max_new: int,
    max_turns: int,
    seed: int,
    enc,
) -> dict[str, Any]:
    legal = 0
    parsed = 0
    rewards: list[float] = []
    seqs: list[list[str]] = []
    recalls: list[float] = []
    leak = 0
    for i, (qid, query) in enumerate(queries):
        _points, _rows, reward, stats = _one_episode(
            backend,
            query=query,
            query_id=qid,
            max_turns=max_turns,
            max_new=max_new,
            policy_version="eval",
            seed=seed + i,
            sample=False,
            enc=enc,
            rollout_idx=0,
        )
        pre = render_student_prompt(snap_from_state(qid, new_state(query, doc_store_for(query))), component_id=COMPONENT)
        if "VERIFY_RESULT_SECRET" in pre or "teacher_verify_judgment" in pre:
            leak += 1
        names = list(stats["names"])
        ok = all(n in STUDENT_NATIVE_TOOLS for n in names) and bool(names)
        parsed += int(ok)
        legal += int(ok)
        rewards.append(reward)
        seqs.append(names)
        recalls.append(float(stats["gold_recall"]))
    n = max(1, len(queries))
    return {
        "n": len(queries),
        "legal_action_rate": legal / n,
        "parse_rate": parsed / n,
        "mean_reward": sum(rewards) / n,
        "mean_gold_recall": sum(recalls) / n,
        "action_sequences": seqs,
        "teacher_leak_rate": leak / n,
    }


def cell_lambda(name: str, lambda_opd: float) -> float:
    if name in {"before", "rl"}:
        return 0.0
    return float(lambda_opd)


def group_stats(groups: list[HybridRolloutGroup]) -> dict[str, Any]:
    n_const = sum(1 for g in groups if len(set(round(r, 6) for r in g.terminal_rewards)) <= 1)
    return {
        "n_groups": len(groups),
        "n_constant_reward_groups": n_const,
        "n_variable_reward_groups": len(groups) - n_const,
        "reward_mean": sum(r for g in groups for r in g.terminal_rewards) / max(1, sum(len(g.terminal_rewards) for g in groups)),
        "tool_seqs": [seq for g in groups for seq in ((g.trajectory_group or {}).get("tool_seqs") or [])],
    }


async def train_cell(
    *,
    name: str,
    backend: ScapeHFToolOPD,
    groups: list[HybridRolloutGroup],
    lambda_opd: float,
    train_steps: int,
    policy_version: str,
    opd_states_per_trajectory: int,
) -> dict[str, Any]:
    if name == "before":
        return {
            "update_type": "eval_only",
            "n_optimizer_steps": 0,
            "n_rl_forward_backward": 0,
            "n_opd_forward_backward": 0,
            "skipped_teacher": True,
        }
    client = HFDebugTrainingClient(backend)
    rl_by_q = {
        g.query_id: list((g.trajectory_group or {}).get("rl_rows") or []) for g in groups
    }
    metrics_acc: list[dict[str, Any]] = []
    last_batch_stats: dict[str, Any] = {}
    teacher = None if lambda_opd <= 0 else teacher_from_point
    for step in range(train_steps):
        batch = prepare_hybrid_batch(
            groups=groups,
            rl_datums_by_query=rl_by_q,
            policy_version=policy_version,
            lambda_opd=lambda_opd,
            component_id=COMPONENT,
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


def main() -> int:
    args = apply_scale(parse_args())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model_src = args.sft_adapter if Path(args.sft_adapter).exists() else args.base_model
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
    queries = [(f"q{i}", q) for i, q in enumerate(expand_queries(args.n_queries))]
    manifest = {
        "training_mode": "four_cell_full" if not args.smoke else "four_cell_smoke",
        "scale": "smoke" if args.smoke else "full",
        "backend": "hf_debug",
        "rl_loss_fn": "cispo",
        "lambda_opd": args.lambda_opd,
        "base_checkpoint": args.base_model,
        "sft_adapter": model_src,
        "target_component": COMPONENT,
        "opd_loss": "sr_opd_ce",
        "opd_state_source": "current_on_policy_rl_rollout",
        "joint_update_contract": "rl_fb+opd_fb+single_optim",
        "legacy_tool_token_kl_hook_used": False,
        "protocol_complete_rl_opd": True,
        "protocol_name": "sr_projected_rl_opd",
        "n_queries": args.n_queries,
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "train_steps": args.train_steps,
        "same_theta0": True,
        "eval_harness": "H_min",
        "local_tool_executor": True,
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cells: dict[str, Any] = {}
    for cell in CELLS:
        restore_trainable(backend.model, theta0)
        backend.optimizer = torch.optim.AdamW(
            [p for p in backend.model.parameters() if p.requires_grad],
            lr=1e-5,
        )
        policy = "v0"
        print(f"[four_cell] cell={cell} rollout", flush=True)
        groups = [
            rollout_group(
                backend,
                query=query,
                query_id=qid,
                group_size=args.group_size,
                max_turns=args.max_turns,
                max_new=args.max_new_tokens,
                policy_version=policy,
                seed=args.seed + 100 * (abs(hash(cell)) % 1000),
                sample=True,
                enc=enc,
            )
            for qid, query in queries
        ]
        gstat = group_stats(groups)
        rewards_before_teacher = [r for g in groups for r in g.terminal_rewards]
        train_stats = asyncio.run(
            train_cell(
                name=cell,
                backend=backend,
                groups=groups,
                lambda_opd=cell_lambda(cell, args.lambda_opd),
                train_steps=0 if cell == "before" else args.train_steps,
                policy_version=policy,
                opd_states_per_trajectory=args.opd_states_per_trajectory,
            )
        )
        rewards_after_teacher = [r for g in groups for r in g.terminal_rewards]
        if rewards_before_teacher != rewards_after_teacher:
            raise RuntimeError("Teacher shadow mutated RL rewards")
        print(f"[four_cell] cell={cell} eval", flush=True)
        ev = eval_cell(
            backend,
            queries,
            max_new=args.max_new_tokens,
            max_turns=args.max_turns,
            seed=args.seed,
            enc=enc,
        )
        cells[cell] = {
            "eval": ev,
            "train": train_stats,
            "rollout": gstat,
            "rollout_reward_mean": sum(rewards_before_teacher) / max(1, len(rewards_before_teacher)),
            "n_decision_points": sum(len(g.decision_points) for g in groups),
            "reward_unchanged_by_teacher": True,
        }
        (out / f"{cell}.json").write_text(json.dumps(cells[cell], indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "cell": cell,
                    **ev,
                    "n_variable_groups": gstat["n_variable_reward_groups"],
                    "rl_fb": train_stats.get("n_rl_forward_backward"),
                    "opd_fb": train_stats.get("n_opd_forward_backward"),
                    "opt": train_stats.get("n_optimizer_steps"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    rl_opd = cells["rl_opd"]["train"]
    summary = {
        "elapsed_sec": time.time() - t0,
        "manifest": manifest,
        "cells": cells,
        "q1_joint_one_optim": (
            int(rl_opd.get("n_rl_forward_backward") or 0) >= 1
            and int(rl_opd.get("n_opd_forward_backward") or 0) >= 1
            and int(rl_opd.get("n_optimizer_steps") or 0) == args.train_steps
        ),
        "q2_on_policy_projection": bool(cells["rl_opd"]["n_decision_points"]),
        "q3_teacher_does_not_change_reward": all(c["reward_unchanged_by_teacher"] for c in cells.values()),
    }
    (out / "FOUR_CELL_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "out": str(out), "elapsed_sec": summary["elapsed_sec"], "q1": summary["q1_joint_one_optim"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
