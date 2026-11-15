"""Rollout collector: Student on-policy episodes -> RL/OPD training rows.

Student trajectories are driven ONLY by the Student's own sampled actions under
H0.  Teacher branches are separate (``opd.pipeline``) and never replace the
sampled actions or share old logprobs.  Outcome reward is the allowed
supervision channel; hidden answer content is never a runtime observation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.events import CostRecord, total_cost
from rsi.contracts.observation import ScoreRecord
from rsi.contracts.task import PublicTask
from rsi.contracts.trace import EpisodeTrace
from rsi.harness.loader import LoadedBank
from rsi.opd.pipeline import DecisionAlignmentRecord, collect_opd_rows
from rsi.opd.records import OPDRow, RLRow, build_rl_row
from rsi.opd.recipes import Recipe
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop


@dataclass
class RolloutBatch:
    traces: tuple[EpisodeTrace, ...]
    rewards: tuple[float, ...]
    rl_rows: tuple[RLRow, ...]
    opd_rows: tuple[OPDRow, ...]
    score_records: tuple[ScoreRecord, ...]
    cost: CostRecord
    wall_ms: float = 0.0
    alignment_records: tuple[DecisionAlignmentRecord, ...] = ()

    @property
    def n_episodes(self) -> int:
        return len(self.traces)


def _empty_trace(task):
    """Minimal trace for a timed-out episode (no decisions, honest marker)."""
    from rsi.contracts.trace import EpisodeTrace

    return EpisodeTrace(
        episode_id=f"timeout-{task.task_id}", task_id=task.task_id,
        domain=task.domain, policy_digest="", bank_digest="",
        events=(), decisions=(), final_answer=None, outcome_reward=None,
        terminated_reason="episode_timeout", synthetic=False,
    )


def _run_episode_child(loop, task, seed, q, task_provider):
    """Child entry for the episode-timeout wrapper (fork context)."""
    import time as _time

    try:
        result = loop.run_episode(task, seed=seed)
        scorer = None
        # scorer is built by the caller; re-derive from the env the loop holds
        env = loop.env
        from rsi.environments.registry import global_registry

        registry = global_registry()
        bundle = registry.get(task.domain)
        scorer = bundle.scorer_factory()
        artifact = result.final_artifact
        if artifact is None:
            from rsi.contracts.observation import ArtifactRef

            artifact = ArtifactRef(ref_id=task.task_id, kind="answer")
        record = scorer.evaluate(artifact, task_provider.private_ref(task.task_id))
        q.put((result, record))
    except Exception as e:  # noqa: BLE001 - child errors surface as None
        try:
            q.put((None, None))
        except Exception:  # noqa: BLE001
            pass


def collect_rollouts(
    *,
    policy: Any,
    teacher_policy: Any,
    env_factory: Any,
    scorer_factory: Any,
    task_provider: Any,
    student_bank: LoadedBank,
    teacher_bank: LoadedBank,
    tasks: Sequence[PublicTask],
    budgets: BudgetSpec | None = None,
    sampling: Any | None = None,
    context_policy: Any | None = None,
    reward_fn=None,
    collect_opd: bool = True,
    recipes: Sequence[Recipe] = (),
    uncertain_realizability: str = "skip",
    max_targets_per_decision: int = 1,
    max_shadow_events: int = 8,
    seed: int = 0,
    campaign_id: str = "train",
    batch_index: int = 0,
    center_advantage: bool = True,
) -> RolloutBatch:
    """Run Student rollouts (H0), score outcomes, build RL rows, align OPD."""
    from rsi.opd.teacher import TeacherForkRunner

    budgets = budgets or BudgetSpec()
    traces: list[EpisodeTrace] = []
    rewards: list[float] = []
    score_records: list[ScoreRecord] = []
    rl_rows: list[RLRow] = []
    opd_rows: list[OPDRow] = []
    alignment_records: list[DecisionAlignmentRecord] = []
    cost = CostRecord()
    t0 = time.perf_counter()

    teacher_runner = (
        TeacherForkRunner(
            env_factory=env_factory,
            policy=teacher_policy,
            bank=teacher_bank,
            max_shadow_events=max_shadow_events,
            sampling=sampling,
            budgets=budgets,
        )
        if collect_opd
        else None
    )

    for i, task in enumerate(tasks):
        env = env_factory()
        loop = MinimalLoop(
            env=env,
            policy=policy,
            bank=student_bank,  # Student runs H0 (+empty bank == H0)
            budgets=budgets,
            sampling=sampling,
            context_policy=context_policy,
            campaign_id=campaign_id,
            batch_index=batch_index,
        )
        # pmhlp/8qsy3 postmortem: a single rank stuck inside run_episode (C-level
        # generate/env stall — zero logs for hours) blocks the 31 other ranks at
        # the next ALLREDUCE until the NCCL watchdog kills the whole job. An
        # episode-level subprocess timeout converts that failure into a reward-0
        # skip: the batch loses one episode, the run survives. Enabled via
        # RSI_EPISODE_TIMEOUT_MINUTES (default 90 — the slowest observed healthy
        # episode is well under 60min).
        import os as _os
        _timeout_min = int(_os.environ.get("RSI_EPISODE_TIMEOUT_MINUTES", "90"))
        if _timeout_min > 0:
            import multiprocessing as _mp

            _ctx = _mp.get_context("fork")
            _q = _ctx.Queue()
            _p = _ctx.Process(target=_run_episode_child,
                              args=(loop, task, seed + i, _q, task_provider), daemon=True)
            _p.start()
            _p.join(timeout=_timeout_min * 60)
            if _p.is_alive():
                _p.terminate()
                _p.join(timeout=30)
                if _p.is_alive():
                    _p.kill()
                print(f"[collector] episode TIMEOUT after {_timeout_min}min "
                      f"(task={task.task_id}) — skipped with reward 0", flush=True)
                from rsi.contracts.observation import ScoreRecord as _SR

                score_records.append(_SR(
                    task_id=task.task_id, domain=task.domain,
                    metric_name="episode_timeout", value=0.0, success=False,
                    details={"measurement_status": "VALID",
                             "error_kind": "episode_timeout",
                             "timeout_minutes": _timeout_min},
                    scorer_version="collector-timeout", synthetic=False,
                ))
                rewards.append(0.0)
                traces.append(_empty_trace(task))
                env.close()
                continue
            result, record = _q.get() if not _q.empty() else (None, None)
            if result is None:
                # child died before putting a result (import/exec failure in
                # fork mode is rare; treat as timeout-equivalent skip)
                print(f"[collector] episode child returned no result "
                      f"(task={task.task_id}) — skipped with reward 0", flush=True)
                from rsi.contracts.observation import ScoreRecord as _SR

                score_records.append(_SR(
                    task_id=task.task_id, domain=task.domain,
                    metric_name="episode_child_failure", value=0.0, success=False,
                    details={"measurement_status": "VALID",
                             "error_kind": "episode_child_failure"},
                    scorer_version="collector-timeout", synthetic=False,
                ))
                rewards.append(0.0)
                traces.append(_empty_trace(task))
                env.close()
                continue
        else:
            result = loop.run_episode(task, seed=seed + i)
        trace = result.trace
        scorer = scorer_factory()
        artifact = result.final_artifact
        if artifact is None:
            from rsi.contracts.observation import ArtifactRef

            artifact = ArtifactRef(ref_id=task.task_id, kind="answer")
        record = scorer.evaluate(artifact, task_provider.private_ref(task.task_id))
        score_records.append(record)
        reward = float(reward_fn(trace, record)) if reward_fn else float(record.value)
        rewards.append(reward)
        traces.append(trace)
        cost = CostRecord(
            actor_input_tokens=cost.actor_input_tokens + total_cost(trace.events).actor_input_tokens,
            actor_output_tokens=cost.actor_output_tokens + total_cost(trace.events).actor_output_tokens,
            tool_calls=cost.tool_calls + total_cost(trace.events).tool_calls,
            wall_ms=cost.wall_ms,
        )
        # RL rows: the Student's real sampled actions with outcome advantage.
        for decision in trace.decisions:
            rl_rows.append(build_rl_row(decision, advantage=reward, reward_ref=record.task_id))
        # OPD rows: Teacher branches aligned to Student-native supervision.
        if collect_opd and teacher_runner is not None:
            records, _stats = collect_opd_rows(
                policy=policy,
                teacher_policy=teacher_policy,
                env=env,
                bank=teacher_bank,
                task=task,
                trace=trace,
                recipes=recipes,
                uncertain_realizability=uncertain_realizability,
                max_targets_per_decision=max_targets_per_decision,
                max_shadow_events=max_shadow_events,
                teacher_runner=teacher_runner,
            )
            alignment_records.extend(records)
            opd_rows.extend(r.opd_row for r in records if r.opd_row is not None)
        env.close()

    # Batch-centered advantage (REINFORCE-style baseline); per-row reward kept
    # in reward_ref for audit.  Distributed callers center later with the
    # GLOBAL mean reward (center_advantage=False).
    if rl_rows and center_advantage:
        mean_r = sum(rewards) / max(1, len(rewards))
        rl_rows = [
            RLRow(
                row_id=r.row_id,
                episode_id=r.episode_id,
                decision_index=r.decision_index,
                prompt_token_ids=r.prompt_token_ids,
                action_token_ids=r.action_token_ids,
                old_logprobs=r.old_logprobs,
                action_mask=r.action_mask,
                advantage=r.advantage - mean_r if len(rewards) > 1 else r.advantage,
                reward_ref=r.reward_ref,
                behavior_policy_version=r.behavior_policy_version,
            )
            for r in rl_rows
        ]
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return RolloutBatch(
        traces=tuple(traces),
        rewards=tuple(rewards),
        rl_rows=tuple(rl_rows),
        opd_rows=tuple(opd_rows),
        score_records=tuple(score_records),
        cost=cost,
        wall_ms=wall_ms,
        alignment_records=tuple(alignment_records),
    )
