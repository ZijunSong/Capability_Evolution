"""Parallel training-rollout collection (multi-GPU).

Each worker pins one GPU, rebuilds the *current* policy snapshot from a base
model + exported LoRA state, runs Student episodes under **H0 only**, runs
Teacher branches under **H0 + champion bank** (all-teacher / zero-student
HarnessOPD), aligns OPD rows inside the worker and returns JSON-serializable
RL/OPD rows.  The parent merges rows and performs the single joint update —
workers never train.

Batch-snapshot semantics: every worker loads the LoRA state exported at batch
start, so Teacher and Student share one frozen snapshot per batch.
Module import must stay torch-free at top level (spawn children pin the device
before the lazy torch import).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Mapping, Sequence

from rsi.contracts.errors import InfraError, ProtocolViolation
from rsi.contracts.events import CostRecord
from rsi.evaluation.parallel import resolve_worker_plan


def _train_rollout_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    gpu_id = payload["gpu_id"]
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from rsi.config import load_config
    from rsi.contracts.observation import ArtifactRef
    from rsi.contracts.policy import SamplingParams
    from rsi.environments.registry import global_registry
    from rsi.harness.loader import load_bank_ref, load_empty_bank
    from rsi.opd.pipeline import collect_opd_rows
    from rsi.opd.records import build_rl_row
    from rsi.providers.factory import build_policy
    from rsi.runtime.factory import build_runtime_parts
    from rsi.runtime.loop import MinimalLoop

    cfg = load_config(payload["config_path"])
    parts = build_runtime_parts(cfg.section("runtime"))
    budgets, sampling = parts.budgets, parts.sampling
    context_policy = parts.context_policy
    device = "cuda:0" if gpu_id is not None else "cpu"
    try:
        policy = build_policy(cfg, device=device, provider_override=payload.get("provider"))
        # load the batch-snapshot LoRA state when present
        state_path = payload.get("lora_state_path")
        if state_path and os.path.isfile(state_path):
            import torch

            from rsi.training.lora import LoRAConfig, apply_lora, load_lora_state_dict

            cfg_lora = dict(cfg.section("training").get("lora") or {})
            apply_lora(
                policy.model,
                LoRAConfig(
                    rank=int(cfg_lora.get("rank", 32)),
                    alpha=float(cfg_lora.get("alpha", 64)),
                    target_suffixes=tuple(cfg_lora.get("target_suffixes") or LoRAConfig().target_suffixes),
                ),
            )
            load_lora_state_dict(policy.model, torch.load(state_path, map_location="cpu", weights_only=True))
        student_bank = load_empty_bank()  # zero student: H0 only
        teacher_bank = load_bank_ref(payload["teacher_bank_ref"])  # all teacher: H0 + Bk
    except Exception as e:  # noqa: BLE001
        return {"error": f"worker setup failed: {e!r}", "rl_rows": [], "opd_rows": [], "episodes": []}

    registry = global_registry()
    out_rows: dict[str, Any] = {"rl_rows": [], "opd_rows": [], "episodes": []}
    collect_opd = bool(payload.get("collect_opd", True))

    for spec in payload["episodes"]:
        try:
            bundle = registry.get(spec["domain"])
            provider = bundle.task_provider_factory()
            scorer = bundle.scorer_factory()
            task = next(t for t in provider.all_tasks() if t.task_id == spec["task_id"])
            env = bundle.adapter_factory()
            loop = MinimalLoop(
                env=env,
                policy=policy,
                bank=student_bank,
                budgets=budgets,
                sampling=sampling,
                context_policy=context_policy,
                campaign_id=payload["campaign_id"],
                batch_index=payload["batch_index"],
            )
            result = loop.run_episode(task, seed=spec["seed"])
            artifact = result.final_artifact or ArtifactRef(ref_id=task.task_id, kind="answer")
            record = scorer.evaluate(artifact, provider.private_ref(task.task_id))
            reward = float(record.value)  # task_outcome reward
            for decision in result.trace.decisions:
                out_rows["rl_rows"].append(
                    build_rl_row(decision, advantage=reward, reward_ref=record.task_id).to_json()
                )
            if collect_opd:
                records, _stats = collect_opd_rows(
                    policy=policy,
                    teacher_policy=policy,  # batch-snapshot: same frozen weights
                    env=env,
                    bank=teacher_bank,
                    task=task,
                    trace=result.trace,
                    uncertain_realizability=str(cfg.section("opd").get("uncertain_realizability", "skip")),
                    max_targets_per_decision=int(cfg.section("opd").get("max_targets_per_decision", 1)),
                    max_shadow_events=int(cfg.section("opd").get("max_shadow_events", 4)),
                )
                out_rows["opd_rows"].extend(r.opd_row.to_json() for r in records if r.opd_row is not None)
            out_rows["episodes"].append(
                {
                    "task_id": record.task_id,
                    "domain": record.domain,
                    "value": record.value,
                    "success": record.success,
                    "reward": reward,
                    "terminated_reason": result.trace.terminated_reason,
                    "n_decisions": len(result.trace.decisions),
                    "component_errors": dict(result.component_errors),
                    "synthetic": record.synthetic,
                    "cost": CostRecord(
                        actor_output_tokens=sum(e.cost.actor_output_tokens for e in result.trace.events),
                        tool_calls=sum(e.cost.tool_calls for e in result.trace.events),
                    ).to_json(),
                }
            )
            env.close()
        except Exception as e:  # noqa: BLE001 - one episode must not kill the batch
            out_rows["episodes"].append(
                {"task_id": spec["task_id"], "domain": spec["domain"], "error": repr(e), "value": 0.0, "reward": 0.0}
            )
    return out_rows


def collect_rollouts_mp(
    cfg: Any,
    *,
    lora_state_path: str | None,
    teacher_bank_ref: str,
    episodes: Sequence[Mapping[str, Any]],
    campaign_id: str,
    batch_index: int,
    workers: int | None = None,
    gpu_ids: Sequence[int] | None = None,
    collect_opd: bool = True,
) -> dict[str, Any]:
    """Collect one on-policy batch across pinned GPU workers.

    ``episodes`` items: {domain, task_id, seed}.  Returns merged
    ``{rl_rows, opd_rows, episodes}`` with rows already rebuilt as objects.
    """
    from rsi.opd.records import OPDRow, RLRow

    if not hasattr(cfg, "config_path"):
        raise ProtocolViolation("collect_rollouts_mp requires a loaded CampaignConfig")
    n_workers, gpu_plan = resolve_worker_plan(workers=workers, gpu_ids=gpu_ids, require_gpu=False)
    if not episodes:
        return {"rl_rows": [], "opd_rows": [], "episodes": []}

    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    chunk_gpu: list[int | None] = [None] * n_workers
    for i, ep in enumerate(episodes):
        w = i % n_workers
        chunks[w].append(dict(ep))
        chunk_gpu[w] = gpu_plan[i % len(gpu_plan)]

    payloads = [
        {
            "config_path": str(cfg.config_path),
            "lora_state_path": lora_state_path,
            "teacher_bank_ref": teacher_bank_ref,
            "campaign_id": campaign_id,
            "batch_index": batch_index,
            "collect_opd": collect_opd,
            "gpu_id": chunk_gpu[w],
            "episodes": chunks[w],
        }
        for w in range(n_workers)
        if chunks[w]
    ]
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(processes=len(payloads)) as pool:
            results = pool.map(_train_rollout_worker, payloads)
    except Exception as e:  # noqa: BLE001
        raise InfraError(f"rollout worker failed: {e!r}") from e

    merged = {"rl_rows": [], "opd_rows": [], "episodes": []}
    errors = []
    for res in results:
        if res.get("error"):
            errors.append(res["error"])
        merged["rl_rows"].extend(RLRow.from_json(r) for r in res.get("rl_rows", []))
        merged["opd_rows"].extend(OPDRow.from_json(r) for r in res.get("opd_rows", []))
        merged["episodes"].extend(res.get("episodes", []))
    if errors and not merged["rl_rows"]:
        raise InfraError(f"all rollout workers failed: {errors[:3]}")
    # batch-centered advantage (REINFORCE-style baseline), same as the serial path
    rewards = [float(e.get("reward", 0.0)) for e in merged["episodes"] if "reward" in e]
    if len(rewards) > 1:
        mean_r = sum(rewards) / len(rewards)
        merged["rl_rows"] = [
            RLRow(
                row_id=r.row_id,
                episode_id=r.episode_id,
                decision_index=r.decision_index,
                prompt_token_ids=r.prompt_token_ids,
                action_token_ids=r.action_token_ids,
                old_logprobs=r.old_logprobs,
                action_mask=r.action_mask,
                advantage=r.advantage - mean_r,
                reward_ref=r.reward_ref,
                behavior_policy_version=r.behavior_policy_version,
            )
            for r in merged["rl_rows"]
        ]
    merged["worker_errors"] = errors
    return merged
