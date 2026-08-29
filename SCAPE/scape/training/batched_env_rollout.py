"""Turn-synchronous batched env rollouts.

All live episodes of a query×group batch share one generate_batch call
per turn so vLLM continuous batching sees hundreds of prompts at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from scape.eval.browsecomp_retrieval import RetrievalBackend
from scape.training.opd_dataset import render_student_prompt
from scape.training.rl_opd_types import HybridRolloutGroup, StudentDecisionPoint
from scape.training.vllm_hybrid import GenerateRequest, GenerateResult, cispo_row_from_generation

GenerateBatch = Callable[[Sequence[GenerateRequest]], list[GenerateResult]]


@dataclass
class LiveEpisode:
    row: dict[str, Any]
    rollout_idx: int
    seed: int
    st: dict[str, Any]
    component_id: str
    policy_version: str
    acts: list[tuple[Any, Any]] = field(default_factory=list)
    points: list[StudentDecisionPoint] = field(default_factory=list)
    rl_rows: list[dict[str, Any]] = field(default_factory=list)
    valids: list[bool] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    pending_pre: Any = None
    pending_prefix: str = ""
    pending_pids: list[int] = field(default_factory=list)
    harness_mask: dict[str, bool] | None = None


def _build_prompt_ids(ep: LiveEpisode, enc) -> list[int]:
    from scape.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids
    from scape.eval.local_search_env import wm_text

    query = str(ep.row["query"])
    if not ep.acts:
        return build_first_turn_prompt_ids(query, enc=enc)
    return build_continuation_prompt_ids(
        query,
        actions_obs=ep.acts,
        wm_text=wm_text(ep.st, auto_on=False),
        enc=enc,
    )


def _apply_generation(
    ep: LiveEpisode,
    gen: GenerateResult,
    *,
    enc,
    searcher: RetrievalBackend | None,
) -> None:
    from scape.eval.harmony_runtime import decode_ids, make_action, make_observation
    from scape.eval.local_search_env import execute_tool
    from scape.training.four_cell_runtime import parse_generated_action, snap_from_state

    qid = str(ep.row["query_id"])
    action, valid = parse_generated_action(gen.text, gen.token_ids, enc)
    ep.valids.append(valid)
    ep.actions.append(action)
    ep.names.append(str(action.get("name")))
    ep.st, obs, _ok = execute_tool(ep.st, action.get("name") if valid else None, action.get("arguments"))
    if valid:
        try:
            ep.acts.append((make_action(action["name"], action.get("arguments") or {}), make_observation(obs)))
        except Exception:
            pass
    action_ids = list(gen.token_ids)
    prompt_ids = list(ep.pending_pids)
    prompt_text = ""
    if enc is not None:
        try:
            prompt_text = decode_ids(enc, prompt_ids)
        except Exception:
            prompt_text = ""
    prompt_text = prompt_text or ep.pending_prefix
    post = snap_from_state(qid, ep.st, ep.component_id, harness_mask=ep.harness_mask)
    ep.points.append(
        StudentDecisionPoint(
            episode_id=f"{qid}_r{ep.rollout_idx}",
            query_id=qid,
            rollout_idx=ep.rollout_idx,
            turn_id=len(ep.points),
            policy_version=ep.policy_version,
            pre_action_snapshot=ep.pending_pre,
            pre_action_snapshot_hash=ep.pending_pre.content_hash(),
            student_model_input=ep.pending_prefix,
            student_action_tokens=action_ids,
            student_action_text=gen.text,
            action_tool_names=[action.get("name") or ""],
            post_action_snapshot=post,
            reward=None,
            structurally_valid=valid,
        )
    )
    rec = cispo_row_from_generation(
        query_id=qid,
        prompt_ids=prompt_ids,
        prompt_text=prompt_text,
        gen=gen,
        policy_version=ep.policy_version,
        turn_id=ep.points[-1].turn_id,
        valid=valid,
    )
    ep.rl_rows.append(rec)


def rollout_queries_batched(
    generate_batch: GenerateBatch,
    rows: Sequence[dict[str, Any]],
    *,
    component_id: str,
    group_size: int,
    max_turns: int,
    max_new: int,
    policy_version: str,
    seed: int,
    sample: bool,
    enc,
    searcher: RetrievalBackend | None = None,
    teacher_mode: bool = False,
    harness_mask: dict[str, bool] | None = None,
) -> list[HybridRolloutGroup]:
    """Batch across queries and group members; step the env between turns."""
    from scape.eval.local_search_env import curated_recall, new_state
    from scape.training.four_cell_runtime import (
        doc_store_for_row,
        snap_from_state,
        terminal_reward,
    )
    from scape.training.hf_rl_opd_client import group_relative_advantages

    episodes: list[LiveEpisode] = []
    for row in rows:
        store = doc_store_for_row(row, searcher)
        for g in range(group_size):
            episodes.append(
                LiveEpisode(
                    row=row,
                    rollout_idx=g,
                    seed=int(seed) + 17 * g,
                    st=new_state(str(row["query"]), store),
                    component_id=component_id,
                    policy_version=policy_version,
                    harness_mask=harness_mask,
                )
            )

    temperature = 1.0 if sample else 0.0
    for turn in range(max_turns):
        live = [ep for ep in episodes if not ep.st.get("ended")]
        if not live:
            break
        reqs: list[GenerateRequest] = []
        generated: list[GenerateResult | None] = [None] * len(live)
        request_slots: list[int] = []
        for i, ep in enumerate(live):
            pids = _build_prompt_ids(ep, enc)
            pre = snap_from_state(str(ep.row["query_id"]), ep.st, component_id, harness_mask=ep.harness_mask)
            ep.pending_pre = pre
            ep.pending_prefix = render_student_prompt(pre, component_id=component_id)
            ep.pending_pids = pids
            if teacher_mode:
                from scape.training.action_codec import render_action
                from scape.training.sentence_compress_teacher import teacher_events_from_point
                from scape.training.rl_opd_types import StudentDecisionPoint

                point = StudentDecisionPoint(
                    episode_id=f"{ep.row['query_id']}_r{ep.rollout_idx}",
                    query_id=str(ep.row["query_id"]),
                    rollout_idx=ep.rollout_idx,
                    turn_id=turn,
                    policy_version=policy_version,
                    pre_action_snapshot=pre,
                    pre_action_snapshot_hash=pre.content_hash(),
                    student_model_input=ep.pending_prefix,
                    student_action_tokens=[],
                    student_action_text="",
                    action_tool_names=[],
                    post_action_snapshot=pre,
                    reward=None,
                    structurally_valid=True,
                )
                action_event = next(e for e in teacher_events_from_point(point) if e.action_name)
                text = render_action({"name": action_event.action_name, "arguments": action_event.arguments})
                token_ids = list(enc.encode(text))
                generated[i] = GenerateResult(
                    request_id=f"{ep.row['query_id']}:g{ep.rollout_idx}:t{turn}:{i}",
                    token_ids=token_ids,
                    token_logprobs=[0.0] * len(token_ids),
                    text=text,
                    logprob_old=0.0,
                    logprob_provenance="teacher_projected_action",
                )
            else:
                request_slots.append(i)
                reqs.append(
                    GenerateRequest(
                        request_id=f"{ep.row['query_id']}:g{ep.rollout_idx}:t{turn}:{i}",
                        prompt_token_ids=pids,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        seed=ep.seed + 31 * turn,
                    )
                )
        if reqs:
            gens = generate_batch(reqs)
            if len(gens) != len(reqs):
                raise RuntimeError(f"generate_batch returned {len(gens)} for {len(reqs)} requests")
            for slot, gen in zip(request_slots, gens):
                generated[slot] = gen
        for ep, gen in zip(live, generated):
            if gen is None:
                raise RuntimeError("missing generation for live episode")
            _apply_generation(ep, gen, enc=enc, searcher=searcher)

    by_q: dict[str, list[LiveEpisode]] = {}
    for ep in episodes:
        by_q.setdefault(str(ep.row["query_id"]), []).append(ep)

    groups: list[HybridRolloutGroup] = []
    for row in rows:
        qid = str(row["query_id"])
        members = by_q.get(qid, [])
        points: list[StudentDecisionPoint] = []
        rl_rows: list[dict[str, Any]] = []
        rewards: list[float] = []
        tool_seqs: list[list[str]] = []
        gold_ids = [str(x) for x in (row.get("gold_docids") or row.get("evidence_docids") or [])]
        query = str(row["query"])
        for ep in members:
            reward = terminal_reward(
                ep.st, query=query, gold_ids=gold_ids, valids=ep.valids, actions=ep.actions
            )
            for point in ep.points:
                point.reward = reward
            for rec in ep.rl_rows:
                rec["reward"] = reward
            points.extend(ep.points)
            rl_rows.extend(ep.rl_rows)
            rewards.append(reward)
            tool_seqs.append(list(ep.names))
            ep.st["gold_recall"] = float(curated_recall(ep.st, gold_ids) or 0.0)
        adv = group_relative_advantages([r["reward"] for r in rl_rows], [r["query_id"] for r in rl_rows])
        for rec, a in zip(rl_rows, adv):
            rec["advantage"] = a
        groups.append(
            HybridRolloutGroup(
                query_id=qid,
                policy_version=policy_version,
                trajectory_group={
                    "rl_rows": rl_rows,
                    "query": row.get("query"),
                    "tool_seqs": tool_seqs,
                    "episode_stats": [
                        {
                            "names": list(ep.names),
                            "reward": ep.points[0].reward if ep.points else 0.0,
                            "n_curated": len(ep.st.get("curated") or {}),
                            "gold_recall": float(ep.st.get("gold_recall") or 0.0),
                            "ended": bool(ep.st.get("ended")),
                            "n_turns": len(ep.names),
                            "n_tool_calls": int(ep.st.get("n_tool_calls") or 0),
                            "n_search_calls": int(ep.st.get("n_search_calls") or 0),
                            "search_query": next(
                                (a.get("arguments", {}).get("query") for a in ep.actions if a.get("name") == "search_corpus"),
                                query,
                            ),
                        }
                        for ep in members
                    ],
                },
                decision_points=points,
                terminal_rewards=rewards,
                metadata={
                    "n_rl_rows": len(rl_rows),
                    "reward_spread": (max(rewards) - min(rewards)) if rewards else 0.0,
                    "batched": True,
                },
            )
        )
    return groups


def traces_from_groups(
    groups: Sequence[HybridRolloutGroup],
    rows: Sequence[dict[str, Any]],
    *,
    searcher: RetrievalBackend | None,
    leak_check_fn: Callable[[str], bool] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    from scape.eval.sr_opd_four_cell_eval import search_metrics

    by_q = {g.query_id: g for g in groups}
    traces: list[dict[str, Any]] = []
    leak = 0
    for row in rows:
        group = by_q.get(str(row["query_id"]))
        stats = {}
        reward = 0.0
        if group is not None:
            ep_stats = list((group.trajectory_group or {}).get("episode_stats") or [])
            stats = ep_stats[0] if ep_stats else {}
            reward = float(stats.get("reward") or (group.terminal_rewards[0] if group.terminal_rewards else 0.0))
            prefix = ""
            if group.decision_points:
                prefix = str(group.decision_points[0].student_model_input or "")
            if leak_check_fn and leak_check_fn(prefix):
                leak += 1
            elif "compressed_teacher_view" in prefix or "VERIFY_RESULT_SECRET" in prefix:
                leak += 1
        search_q = str(stats.get("search_query") or row["query"])
        sm = search_metrics(searcher, search_q, list(row.get("evidence_docids") or [])) if searcher is not None else {}
        traces.append(
            {
                "query_id": row["query_id"],
                "tool_names": list(stats.get("names") or []),
                "reward": reward,
                "gold_recall": stats.get("gold_recall"),
                "n_tool_calls": stats.get("n_tool_calls"),
                "n_search_calls": stats.get("n_search_calls"),
                **sm,
            }
        )
    return traces, leak
