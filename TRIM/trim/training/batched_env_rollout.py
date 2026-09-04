"""Turn-synchronous batched env rollouts.

All live episodes of a query×group batch share one generate_batch call
per turn so vLLM continuous batching sees hundreds of prompts at once.

Document stores are prepared in query micro-batches (default: enough
queries to fill ~256 live episodes). The first generate_batch is submitted
as soon as the first micro-batch is ready, and the next batch is prepared
on a background thread while the GPU is busy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import time

from trim.eval.browsecomp_retrieval import RetrievalBackend
from trim.eval.harness1_metrics import EpisodeTiming, episode_quality_metrics, timed_section, trace_fields
from trim.training.opd_dataset import render_student_prompt
from trim.training.rl_opd_types import HybridRolloutGroup, StudentDecisionPoint
from trim.training.vllm_hybrid import GenerateRequest, GenerateResult, cispo_row_from_generation

GenerateBatch = Callable[[Sequence[GenerateRequest]], list[GenerateResult]]

# Keep about this many live episodes in one vLLM generate_batch.
# With --group-size 8 that is 32 queries; eval (group_size=1) gets 256.
DEFAULT_TARGET_LIVE_EPISODES = 256
DEFAULT_DOC_STORE_WORKERS = 8


def resolved_query_batch_size(
    n_rows: int,
    group_size: int,
    query_batch_size: int | None,
) -> int:
    """How many queries to prepare before the first generate_batch call."""
    n_rows = max(0, int(n_rows))
    if query_batch_size is not None:
        n = int(query_batch_size)
        if n <= 0:
            return max(1, n_rows) if n_rows else 1
        return max(1, n)
    target = max(1, int(DEFAULT_TARGET_LIVE_EPISODES) // max(1, int(group_size)))
    if n_rows:
        return max(1, min(n_rows, target))
    return target


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
    timing: EpisodeTiming = field(default_factory=EpisodeTiming)


def _build_prompt_ids(ep: LiveEpisode, enc) -> list[int]:
    from trim.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids
    from trim.eval.local_search_env import wm_text

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
    search_k: int = 10,
) -> None:
    from trim.eval.harmony_runtime import decode_ids, make_action, make_observation
    from trim.eval.local_search_env import execute_tool
    from trim.training.four_cell_runtime import parse_generated_action, snap_from_state

    qid = str(ep.row["query_id"])
    action, valid = parse_generated_action(gen.text, gen.token_ids, enc)
    ep.valids.append(valid)
    ep.actions.append(action)
    ep.names.append(str(action.get("name")))
    with timed_section(ep.timing, "harness"):
        ep.st, obs, _ok = execute_tool(
            ep.st,
            action.get("name") if valid else None,
            action.get("arguments"),
            searcher=searcher,
            search_k=search_k,
        )
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
            student_prompt_token_ids=list(prompt_ids),
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


def _prepare_chunk_episodes(
    chunk: Sequence[dict[str, Any]],
    *,
    component_id: str,
    group_size: int,
    policy_version: str,
    seed: int,
    harness_mask: dict[str, bool] | None,
    searcher: RetrievalBackend | None,
    doc_store_k: int,
    doc_store_workers: int,
    new_state,
    doc_store_for_row,
) -> list[LiveEpisode]:
    workers = max(1, int(doc_store_workers or 1))
    if workers == 1 or len(chunk) <= 1:
        stores = [doc_store_for_row(row, searcher, k=doc_store_k) for row in chunk]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as pool:
            stores = list(
                pool.map(lambda row: doc_store_for_row(row, searcher, k=doc_store_k), chunk)
            )
    episodes: list[LiveEpisode] = []
    for row, store in zip(chunk, stores):
        copied = dict(store)
        for g in range(group_size):
            episodes.append(
                LiveEpisode(
                    row=row,
                    rollout_idx=g,
                    seed=int(seed) + 17 * g,
                    st=new_state(str(row["query"]), dict(copied)),
                    component_id=component_id,
                    policy_version=policy_version,
                    harness_mask=harness_mask,
                )
            )
    return episodes


def _run_episode_turns(
    episodes: list[LiveEpisode],
    generate_batch: GenerateBatch,
    *,
    component_id: str,
    max_turns: int,
    max_new: int,
    policy_version: str,
    enc,
    searcher: RetrievalBackend | None,
    teacher_mode: bool,
    temperature: float,
    search_k: int,
    snap_from_state,
) -> None:
    for turn in range(max_turns):
        live = [ep for ep in episodes if not ep.st.get("ended")]
        if not live:
            break
        reqs: list[GenerateRequest] = []
        generated: list[GenerateResult | None] = [None] * len(live)
        request_slots: list[int] = []
        for i, ep in enumerate(live):
            with timed_section(ep.timing, "harness"):
                pids = _build_prompt_ids(ep, enc)
                pre = snap_from_state(str(ep.row["query_id"]), ep.st, component_id, harness_mask=ep.harness_mask)
                ep.pending_pre = pre
                ep.pending_prefix = render_student_prompt(pre, component_id=component_id)
                ep.pending_pids = pids
            if teacher_mode:
                from trim.training.action_codec import render_action
                from trim.training.sentence_compress_teacher import teacher_events_from_point
                from trim.training.rl_opd_types import StudentDecisionPoint

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
            t_gen = time.perf_counter()
            gens = generate_batch(reqs)
            gen_dt = time.perf_counter() - t_gen
            share = gen_dt / max(1, len(reqs))
            if len(gens) != len(reqs):
                raise RuntimeError(f"generate_batch returned {len(gens)} for {len(reqs)} requests")
            for slot, gen in zip(request_slots, gens):
                generated[slot] = gen
                live[slot].timing.add_model(share)
        for ep, gen in zip(live, generated):
            if gen is None:
                raise RuntimeError("missing generation for live episode")
            _apply_generation(ep, gen, enc=enc, searcher=searcher, search_k=search_k)


def _groups_from_episodes(
    episodes: list[LiveEpisode],
    rows: Sequence[dict[str, Any]],
    *,
    policy_version: str,
    max_turns: int,
    terminal_reward,
    curated_recall,
    group_relative_advantages,
) -> list[HybridRolloutGroup]:
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
        episode_stats: list[dict[str, Any]] = []
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
            quality = episode_quality_metrics(
                ep.st,
                row,
                tool_names=ep.names,
                valids=ep.valids,
                reward=reward,
                max_turns=max_turns,
                timing=ep.timing.snapshot(),
                actions=ep.actions,
            )
            quality.update(
                {
                    "names": list(ep.names),
                    "ended": bool(ep.st.get("ended")),
                    "n_turns": len(ep.names),
                    "n_tool_calls": int(ep.st.get("n_tool_calls") or 0),
                    "n_search_calls": int(ep.st.get("n_search_calls") or 0),
                    "search_query": quality.get("search_query") or query,
                }
            )
            episode_stats.append(quality)
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
                    "episode_stats": episode_stats,
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
    temperature: float | None = None,
    search_k: int = 10,
    doc_store_k: int = 12,
    query_batch_size: int | None = None,
    doc_store_workers: int = DEFAULT_DOC_STORE_WORKERS,
) -> list[HybridRolloutGroup]:
    """Batch across queries and group members; step the env between turns.

    Document-store prep is chunked so the first vLLM generate_batch runs as
    soon as one micro-batch is ready. The next chunk is prepared on a
    background thread while the GPU rolls out the current chunk.
    """
    from trim.eval.local_search_env import curated_recall, new_state
    from trim.training.four_cell_runtime import (
        doc_store_for_row,
        snap_from_state,
        terminal_reward,
    )
    from trim.training.hf_rl_opd_client import group_relative_advantages

    rows = list(rows)
    if not rows:
        return []
    batch = resolved_query_batch_size(len(rows), group_size, query_batch_size)
    workers = max(1, int(doc_store_workers or 1))
    chunks = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    temperature = 0.0 if not sample else float(temperature if temperature is not None else 1.0)

    def prepare(chunk: Sequence[dict[str, Any]]) -> tuple[list[LiveEpisode], float]:
        t0 = time.perf_counter()
        episodes = _prepare_chunk_episodes(
            chunk,
            component_id=component_id,
            group_size=group_size,
            policy_version=policy_version,
            seed=seed,
            harness_mask=harness_mask,
            searcher=searcher,
            doc_store_k=doc_store_k,
            doc_store_workers=workers,
            new_state=new_state,
            doc_store_for_row=doc_store_for_row,
        )
        return episodes, time.perf_counter() - t0

    groups: list[HybridRolloutGroup] = []
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        next_fut = prefetch.submit(prepare, chunks[0])
        for i, chunk in enumerate(chunks):
            episodes, prep_s = next_fut.result()
            if i + 1 < len(chunks):
                next_fut = prefetch.submit(prepare, chunks[i + 1])
            if len(chunks) > 1:
                print(
                    f"[rollout] chunk {i + 1}/{len(chunks)} queries={len(chunk)} "
                    f"episodes={len(episodes)} prep={prep_s:.1f}s -> vLLM",
                    flush=True,
                )
            _run_episode_turns(
                episodes,
                generate_batch,
                component_id=component_id,
                max_turns=max_turns,
                max_new=max_new,
                policy_version=policy_version,
                enc=enc,
                searcher=searcher,
                teacher_mode=teacher_mode,
                temperature=temperature,
                search_k=search_k,
                snap_from_state=snap_from_state,
            )
            groups.extend(
                _groups_from_episodes(
                    episodes,
                    chunk,
                    policy_version=policy_version,
                    max_turns=max_turns,
                    terminal_reward=terminal_reward,
                    curated_recall=curated_recall,
                    group_relative_advantages=group_relative_advantages,
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
    from trim.eval.sr_opd_four_cell_eval import search_metrics

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
                **trace_fields(stats),
                **sm,
            }
        )
    return traces, leak
