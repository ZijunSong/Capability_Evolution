"""CLI service for HarnessOPD internalization (`rsi.cli train`).

All-teacher / zero-student wiring: Student rollouts run **H0 only** (empty
bank); Teacher branches run **H0 + the champion bank** inside the collector.

Three execution modes:

- **distributed** (torchrun, WORLD_SIZE>1): every rank collects its shard of
  the on-policy batch on its own GPU, all_reduces global normalizers / mean
  reward / SUM gradients, and takes one identical optimizer step (LoRA-only).
- **pool** (single node, RSI_PANEL_WORKERS>=2): one trainer process +
  pinned-GPU rollout workers (batch-snapshot LoRA state exported per batch).
- **serial**: in-process collection (tests / smoke).

Main path trains a **cumulative LoRA adapter** (base frozen) — full-parameter
AdamW on a 9B model does not fit one 80GB device.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.common import digest_of
from rsi.contracts.errors import ProtocolViolation
from rsi.harness.loader import load_bank_ref, load_empty_bank
from rsi.providers.factory import build_policy
from rsi.training.backend import GlobalNormalizers
from rsi.training.checkpoint import CheckpointIdentity, save_checkpoint
from rsi.training.export import export_adapter
from rsi.training.joint import JointStepReport, MethodSpec, joint_update


def _build_backend(cfg: CampaignConfig, policy: Any, device: str | None = None):
    training = cfg.section("training")
    backend_name = str(training.get("backend", "hf_reference"))
    lambda_opd = float(cfg.section("opd").get("lambda", 0.01))
    gate_beta = float(cfg.section("opd").get("gate_beta", 5.0))
    lr = float(training.get("learning_rate", 1e-4))
    if backend_name == "hf_reference":
        from rsi.training.hf_reference import HFReferenceBackend

        model = getattr(policy, "model", None)
        if model is None:
            raise ProtocolViolation("hf_reference backend requires a policy with a torch model")
        kwargs = dict(lambda_opd=lambda_opd, gate_beta=gate_beta, lr=lr, source_policy=policy)
        if device:
            kwargs["device"] = device
        return HFReferenceBackend(model=model, **kwargs)
    if backend_name == "torch_fsdp2":
        from rsi.training.fsdp2_backend import FSDP2Backend

        model = getattr(policy, "model", None)
        if model is None:
            raise ProtocolViolation("torch_fsdp2 backend requires a policy with a torch model")
        return FSDP2Backend(model=model, lambda_opd=lambda_opd, gate_beta=gate_beta, lr=lr)
    raise ProtocolViolation(f"unknown training backend {backend_name!r}")


def _apply_lora_if_configured(cfg: CampaignConfig, policy: Any) -> dict[str, Any]:
    """Cumulative-adapter main path: attach LoRA and freeze the base model."""
    training = cfg.section("training")
    if str(training.get("adapter_mode", "cumulative")) != "cumulative":
        return {"lora": False}
    from rsi.training.lora import LoRAConfig, apply_lora, freeze_non_lora_, lora_param_count

    lora_cfg = dict(training.get("lora") or {})
    cfg_lora = LoRAConfig(
        rank=int(lora_cfg.get("rank", 32)),
        alpha=float(lora_cfg.get("alpha", 64)),
        target_suffixes=tuple(lora_cfg.get("target_suffixes") or LoRAConfig().target_suffixes),
    )
    replaced = apply_lora(policy.model, cfg_lora)
    if not replaced:
        raise ProtocolViolation("LoRA found no matching Linear modules — check target_suffixes")
    n_trainable = freeze_non_lora_(policy.model)
    return {
        "lora": True,
        "rank": cfg_lora.rank,
        "alpha": cfg_lora.alpha,
        "n_modules": len(replaced),
        "n_trainable_tensors": n_trainable,
        "n_trainable_params": lora_param_count(policy.model),
        "target_suffixes": list(cfg_lora.target_suffixes),
    }


def _build_episode_specs(cfg: CampaignConfig, step: int, total: int) -> list[dict[str, Any]]:
    """Deterministic global episode list of exactly ``total`` specs.

    Identical on all ranks (same seed) and shardable with no idle ranks:
    domains round-robin, tasks cycle through each shuffled pool.
    """
    from rsi.environments.registry import global_registry

    registry = global_registry()
    domains = list(cfg.section("suite").get("domains") or [])
    pools: dict[str, list[Any]] = {}
    for domain in domains:
        provider = registry.get(domain).task_provider_factory()
        pool = list(provider.public_tasks("work_train"))
        if pool:
            pools[domain] = pool
    if not pools or total <= 0:
        return []
    rng = random.Random(cfg.seed * 1000 + step)
    order = {d: [i for i in range(len(pools[d]))] for d in pools}
    for d in order:
        rng.shuffle(order[d])
    cursor = {d: 0 for d in pools}
    cycle = sorted(pools)  # deterministic domain round-robin
    specs: list[dict[str, Any]] = []
    k = 0
    while len(specs) < total:
        d = cycle[k % len(cycle)]
        k += 1
        idxs = order[d]
        j = idxs[cursor[d] % len(idxs)]
        cursor[d] += 1
        specs.append(
            {
                "domain": d,
                "task_id": pools[d][j].task_id,
                "seed": cfg.seed + 1000 * step + len(specs),
            }
        )
    return specs


def run_train(cfg: CampaignConfig, *, bank_ref: str, policy: Any | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    from rsi.training.distributed import (
        DistInfo,
        all_reduce_grads,
        all_reduce_sum,
        barrier,
        init_from_env,
        shard_items,
    )

    dist_info: DistInfo = init_from_env()
    bank = load_bank_ref(bank_ref)  # Teacher bank (H0 + Bk) == "all teacher"
    student_bank = load_empty_bank()  # "zero student": H0 only
    # PR-09 (guide §8.2): build the runtime parts ONCE, before the execution-
    # mode branches.  The distributed branch previously referenced
    # ``runtime_cfg`` / ``_parts`` assigned only in the serial branch — an
    # UnboundLocalError on the first distributed episode, which the branch's
    # broad ``except Exception`` converted into a silent reward-0 stream.
    # Pool workers rebuild from the same raw section and must verify the
    # contract digest matches (collector_parallel does the digest check).
    runtime_cfg = cfg.section("runtime")
    from rsi.runtime.factory import build_runtime_parts

    runtime_parts = build_runtime_parts(runtime_cfg)
    # PR-05 (guide §5.1): the formal-training gate.  A bank is not a teacher
    # merely because an evolve run produced it — require a READY readiness
    # artifact bound to the CURRENT identities before any optimizer step.
    training_cfg_gate = cfg.section("training")
    if bool(training_cfg_gate.get("require_teacher_readiness", False)):
        from rsi.opd.readiness import check_readiness_for_training, load_readiness

        readiness_ref = training_cfg_gate.get("readiness_ref")
        readiness = load_readiness(readiness_ref) if readiness_ref else None
        if readiness is not None and readiness.domain not in (None, "", "*"):
            _gate_domains = list(cfg.section("suite").get("domains") or [])
            if _gate_domains and readiness.domain not in _gate_domains:
                raise ProtocolViolation(
                    f"readiness artifact covers domain {readiness.domain!r} but the "
                    f"training config targets {_gate_domains!r}"
                )
        _gate_env_digest = ""
        _gate_domains2 = list(cfg.section("suite").get("domains") or [])
        if _gate_domains2:
            from rsi.contracts.common import digest_of as _d
            from rsi.environments.registry import global_registry as _gr

            _gate_env_digest = _d(
                _gr().get(_gate_domains2[0]).adapter_factory().environment_identity()
            )
        problems = check_readiness_for_training(
            readiness=readiness,
            bank_digest=bank.bank_digest,
            policy_digest=policy.policy_artifact.policy_id,
            runtime_contract_digest=runtime_parts.profile.digest(),
            environment_digest=_gate_env_digest,
        )
        if problems:
            raise ProtocolViolation(
                "teacher readiness gate FAILED — formal OPD training refused:\n- "
                + "\n- ".join(problems)
            )
    policy = (
        policy
        if policy is not None
        else build_policy(cfg, device=dist_info.device if dist_info.is_distributed else os.environ.get("RSI_TRAIN_DEVICE") or None)
    )
    lora_info = _apply_lora_if_configured(cfg, policy)
    backend = _build_backend(cfg, policy, device=dist_info.device if dist_info.is_distributed else None)
    spec = MethodSpec.resolve(str(cfg.section("training").get("method", "harness_opd_rl")))
    if spec.gate_mode == "none":
        backend.gate_mode = "none"
    if dist_info.is_distributed:
        backend.grad_sync_hook = lambda: all_reduce_grads(policy.model, dist_info)

    runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
    assert runs_root is not None
    run_dir = Path(out_dir) if out_dir else runs_root / cfg.campaign_id / "train"
    if dist_info.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    from rsi.environments.registry import global_registry

    registry = global_registry()
    # PR-09 (guide §8.4): checkpoints previously wrote EMPTY
    # runtime_contract_digest / environment_digest — an unversionable
    # checkpoint identity.  Compute both from the real runtime parts and the
    # first configured domain's environment identity.
    from rsi.runtime.loop import build_runtime_contract

    _domains = list(cfg.section("suite").get("domains") or [])
    _env_identity = registry.get(_domains[0]).adapter_factory().environment_identity() if _domains else {}
    runtime_contract_digest = digest_of(
        {
            "runtime_parts": {
                "max_actions": runtime_parts.budgets.max_actions,
                "max_new_tokens": runtime_parts.sampling.max_new_tokens,
                "temperature": runtime_parts.sampling.temperature,
                "max_chars": runtime_parts.context_policy.max_chars,
                "profile": runtime_parts.profile.digest() if runtime_parts.profile else "",
            },
        }
    )
    environment_digest = digest_of(_env_identity)
    training_cfg = cfg.section("training")
    max_steps = int(training_cfg.get("max_optimizer_steps_per_round", 2))
    per_rank = int(training_cfg.get("episodes_per_rank_per_step", 2))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 10))
    panel_workers = int(
        os.environ.get("RSI_PANEL_WORKERS") or cfg.section("execution").get("panel_workers", 1) or 1
    )

    step_reports: list[JointStepReport] = []
    reward_trace: list[float] = []
    episode_log: list[dict[str, Any]] = []
    global_step = 0

    for step in range(max_steps):
        if dist_info.is_distributed:
            total_eps = per_rank * dist_info.world_size
        elif panel_workers >= 2:
            total_eps = per_rank * panel_workers
        else:
            total_eps = per_rank * max(1, len(cfg.section("suite").get("domains") or []))
        ep_specs = _build_episode_specs(cfg, step, total_eps)

        if dist_info.is_distributed:
            # each rank collects its shard in-process on its own GPU
            my_specs = shard_items(ep_specs, dist_info.rank, dist_info.world_size)
            all_rl: list[Any] = []
            all_opd: list[Any] = []
            local_rewards: list[float] = []
            for spec_ep in my_specs:
                try:
                    bundle = registry.get(spec_ep["domain"])
                    provider = bundle.task_provider_factory()
                    task = next(t for t in provider.all_tasks() if t.task_id == spec_ep["task_id"])
                    from rsi.training.collector import collect_rollouts

                    batch = collect_rollouts(
                    policy=policy,
                    teacher_policy=policy,
                    env_factory=bundle.adapter_factory,
                    scorer_factory=bundle.scorer_factory,
                    task_provider=provider,
                    student_bank=student_bank,
                    teacher_bank=bank,
                    tasks=[task],
                    budgets=runtime_parts.budgets,
                    sampling=runtime_parts.sampling,
                    context_policy=runtime_parts.context_policy,
                    collect_opd=spec.use_opd,
                    uncertain_realizability=str(cfg.section("opd").get("uncertain_realizability", "skip")),
                    max_targets_per_decision=int(cfg.section("opd").get("max_targets_per_decision", 1)),
                    max_shadow_events=int(cfg.section("opd").get("max_shadow_events", 4)),
                    seed=int(spec_ep["seed"]),
                    campaign_id=cfg.campaign_id,
                    batch_index=step,
                    center_advantage=False,
                )
                    all_rl.extend(batch.rl_rows)
                    all_opd.extend(batch.opd_rows)
                    local_rewards.extend(batch.rewards)
                except (NameError, UnboundLocalError, ProtocolViolation, KeyError, TypeError, AttributeError) as e:
                    # PR-09 (guide §8.2): programming/config errors are NOT
                    # episode outcomes.  Swallowing them produced the "job
                    # running, zero valid training data" failure mode.
                    raise
                except Exception as e:  # noqa: BLE001 - one bad episode must not
                    # kill the rank (which would tear down the whole 32-rank job)
                    local_rewards.append(0.0)
                    if dist_info.is_main:
                        print(f"[train] rank0 episode error (recorded, continuing): {e!r}", flush=True)

            # one global sync per batch: normalizers + mean reward
            norm_local = GlobalNormalizers.from_rows(all_rl, all_opd, world_size=1)
            g_tokens, g_opd_w, g_rsum, g_rcount = all_reduce_sum(
                [
                    float(norm_local.global_action_tokens),
                    float(norm_local.global_opd_weight),
                    float(sum(local_rewards)),
                    float(len(local_rewards)),
                ],
                dist_info,
            )
            global_norms = GlobalNormalizers(
                global_action_tokens=max(1, int(g_tokens)),
                global_opd_weight=float(g_opd_w),
                world_size=1,
            )
            mean_r = (g_rsum / g_rcount) if g_rcount > 0 else 0.0
            if g_rcount > 1:
                from rsi.opd.records import RLRow

                all_rl = [
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
                    for r in all_rl
                ]
            report = joint_update(
                backend,
                rl_rows=all_rl,
                opd_rows=all_opd,
                spec=spec,
                lambda_opd=float(cfg.section("opd").get("lambda", 0.01)),
                normalizers=global_norms,
            )
            if dist_info.is_main:
                reward_trace.extend(local_rewards)
                episode_log.append({"step": step, "rank0_episodes": len(local_rewards), "n_rl": len(all_rl), "n_opd": len(all_opd)})
            barrier(dist_info)
        elif panel_workers >= 2:
            # single-node pool mode (rollout workers on pinned GPUs)
            from rsi.training.collector_parallel import collect_rollouts_mp

            lora_state_path = None
            if lora_info.get("lora"):
                import torch

                from rsi.training.lora import lora_state_dict

                snap_dir = run_dir / "batches" / f"step_{step:04d}"
                snap_dir.mkdir(parents=True, exist_ok=True)
                lora_state_path = str(snap_dir / "lora_state.pt")
                torch.save(lora_state_dict(policy.model), lora_state_path)
            merged = collect_rollouts_mp(
                cfg,
                lora_state_path=lora_state_path,
                teacher_bank_ref=bank_ref,
                episodes=ep_specs,
                campaign_id=cfg.campaign_id,
                batch_index=step,
                workers=panel_workers,
                collect_opd=spec.use_opd,
            )
            report = joint_update(
                backend,
                rl_rows=merged["rl_rows"],
                opd_rows=merged["opd_rows"],
                spec=spec,
                lambda_opd=float(cfg.section("opd").get("lambda", 0.01)),
            )
            reward_trace.extend(float(e.get("reward", 0.0)) for e in merged["episodes"])
            episode_log.extend(merged["episodes"])
        else:
            # serial in-process mode
            from rsi.training.collector import collect_rollouts

            all_rl = []
            all_opd = []
            for domain in {s["domain"] for s in ep_specs}:
                bundle = registry.get(domain)
                provider = bundle.task_provider_factory()
                tasks = [
                    next(t for t in provider.all_tasks() if t.task_id == e["task_id"])
                    for e in ep_specs
                    if e["domain"] == domain
                ]
                batch = collect_rollouts(
                    policy=policy,
                    teacher_policy=policy,
                    env_factory=bundle.adapter_factory,
                    scorer_factory=bundle.scorer_factory,
                    task_provider=provider,
                    student_bank=student_bank,
                    teacher_bank=bank,
                    tasks=tasks,
                    budgets=runtime_parts.budgets,
                    sampling=runtime_parts.sampling,
                    context_policy=runtime_parts.context_policy,
                    collect_opd=spec.use_opd,
                    uncertain_realizability=str(cfg.section("opd").get("uncertain_realizability", "skip")),
                    max_targets_per_decision=int(cfg.section("opd").get("max_targets_per_decision", 1)),
                    max_shadow_events=int(cfg.section("opd").get("max_shadow_events", 4)),
                    seed=cfg.seed + 1000 * step,
                    campaign_id=cfg.campaign_id,
                    batch_index=step,
                )
                all_rl.extend(batch.rl_rows)
                all_opd.extend(batch.opd_rows)
                reward_trace.extend(batch.rewards)
            report = joint_update(
                backend,
                rl_rows=all_rl,
                opd_rows=all_opd,
                spec=spec,
                lambda_opd=float(cfg.section("opd").get("lambda", 0.01)),
            )

        step_reports.append(report)
        global_step += 1
        if dist_info.is_main and checkpoint_every > 0 and global_step % checkpoint_every == 0:
            save_checkpoint(
                run_dir / f"step_{global_step:04d}",
                backend=backend,
                identity=CheckpointIdentity(
                    policy_digest=policy.policy_artifact.policy_id,
                    bank_digest=bank.bank_digest,
                    runtime_contract_digest=runtime_contract_digest,
                    environment_digest=environment_digest,
                    data_digest=cfg.digest(),
                    global_step=global_step,
                    batch_identity=f"step={global_step}",
                    method=spec.method,
                ),
            )
            (run_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "global_step": global_step,
                        "last_reward_mean": (sum(reward_trace[-64:]) / max(1, len(reward_trace[-64:]))),
                        "n_opd_rows_last": step_reports[-1].n_opd_rows,
                        "n_rl_rows_last": step_reports[-1].n_rl_rows,
                    },
                    indent=2,
                ),
                "utf-8",
            )

    payload: dict[str, Any] = {"ok": True}
    if dist_info.is_main:
        identity = CheckpointIdentity(
            policy_digest=policy.policy_artifact.policy_id,
            bank_digest=bank.bank_digest,
            runtime_contract_digest=runtime_contract_digest,
            environment_digest=environment_digest,
            data_digest=cfg.digest(),
            global_step=global_step,
            batch_identity=f"steps={global_step}",
            method=spec.method,
        )
        ckpt_dir = save_checkpoint(run_dir / f"step_{global_step:04d}", backend=backend, identity=identity)
        try:
            exported = export_adapter(
                backend, run_dir / "export", parent_policy_id=policy.policy_artifact.policy_id
            )
            export_ok = True
            export_error = None
        except Exception as e:  # noqa: BLE001
            export_ok = False
            export_error = repr(e)
            exported = {}
        payload = {
            "ok": export_ok,
            "command": "train",
            "method": spec.method,
            "backend": backend.name,
            "student_harness": "H0_only",
            "teacher_harness": "H0_plus_bank",
            "bank_ref": bank_ref,
            "bank_digest": bank.bank_digest,
            "policy_digest_in": policy.policy_artifact.policy_id,
            "lora": lora_info,
            "n_trainable_params": getattr(backend, "n_trainable_params", None),
            "world_size": dist_info.world_size,
            "global_steps": global_step,
            "step_reports": [r.to_json() for r in step_reports],
            "reward_trace": reward_trace,
            "episodes": episode_log[:200],
            "checkpoint": str(ckpt_dir),
            "export": exported,
            "export_error": export_error,
        }
        out_path = run_dir / "train_result.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
        payload["result_path"] = str(out_path)
    return payload
