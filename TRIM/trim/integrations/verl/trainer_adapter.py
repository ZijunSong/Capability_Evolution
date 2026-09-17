"""Single-node 8-GPU TRIM RL: per-rank vLLM TP=1 rollout + FSDP2 CISPO update.

This is the T3 first slice. SEC env / sampler / reward stay in TRIM.
FSDP2 owns the actor shard and the single optimizer.step. Vendored verl 0.5
is not imported; CISPO clip mapping matches verl v0.9.0.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from trim.integrations.verl.batch_adapter import (
    drop_constant_reward_groups,
    shard_training_rows,
    training_rows_from_groups,
)
from trim.integrations.verl.joint_objective import resolved_cispo_config
from trim.training.dist_runtime import (
    all_gather_via_disk,
    barrier,
    broadcast_object,
    init_dist_if_needed,
    is_coordinator,
    pin_local_cuda_device,
    shard_for_rank,
)
from trim.training.rl_opd_types import TRAINING_MODE_RL, TRAINING_MODE_RL_OPD


VERL_METHODS = {"rl", "rl+opd"}


def require_verl_trainer() -> None:
    return


def _unsupported_method(method: str) -> None:
    if str(method) not in VERL_METHODS:
        raise SystemExit(
            f"--training-backend verl/fsdp2 currently supports {sorted(VERL_METHODS)}; "
            f"got {method!r}. scape+rl / trim stay on hf_debug until T4."
        )


def _write_resolved_config(out: Path, payload: dict[str, Any]) -> None:
    if is_coordinator():
        (out / "RESOLVED_CONFIG.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_verl_fsdp2_train(args: Any) -> dict[str, Any]:
    """On-policy loop used by ``run_train.py --training-backend verl``."""
    os.environ.setdefault("TRIM_DIST_BACKEND", "nccl")
    os.environ["TRIM_GPU_KEEPALIVE"] = "0"
    pin_local_cuda_device()
    dist = init_dist_if_needed(backend="nccl")
    _unsupported_method(str(getattr(args, "train_method", None) or getattr(args, "training_mode", "")))

    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.batched_env_rollout import rollout_queries_batched
    from trim.training.four_cell_runtime import (
        cell_lambda,
        coerce_runtime_args,
        collection_mode_for_cell,
        open_train_retrieval,
        resolved_rollout_mask,
        resolve_queries,
        teacher_for,
        training_output_occupied,
    )
    from trim.training.rl_opd_metrics import reward_parts_group_stats
    from trim.training.tinker_rl_opd_trainer import HybridLoopState, prepare_hybrid_batch
    from trim.training.train_checkpoint import (
        load_rng_state,
        load_training_resume,
        publish_step_checkpoint,
        save_rng_state,
    )
    from trim.training.train_query_sampler import QuerySampler, QuerySamplerState
    from trim.training.vllm_hybrid import VLLMGenerateClient, materialize_vllm_base, wait_gpus_quiet

    args = coerce_runtime_args(args)
    method = str(args.train_method if hasattr(args, "train_method") else args.training_mode)
    if method in {"rl_opd", TRAINING_MODE_RL_OPD}:
        method = "rl+opd"
    if method == TRAINING_MODE_RL:
        method = "rl"
    _unsupported_method(method)

    out = Path(args.out)
    occupy_err = None
    if is_coordinator():
        if training_output_occupied(out) and not bool(getattr(args, "resume", False)):
            occupy_err = f"output dir {out} already has training state. Pass --resume or use a new --out."
        else:
            out.mkdir(parents=True, exist_ok=True)
    occupy_err = broadcast_object(occupy_err)
    if occupy_err:
        raise SystemExit(occupy_err)
    barrier()
    out.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows, pool_meta = resolve_queries(args)
    del eval_rows
    train_searcher = open_train_retrieval(args, train_rows)
    enc = load_model_encoding(str(args.base_model or args.model_name))
    vllm_base = str(args.base_model or args.model_name)
    if getattr(args, "sft_adapter", None):
        if is_coordinator():
            vllm_base = materialize_vllm_base(
                base_model=vllm_base,
                sft_adapter=str(args.sft_adapter),
                cache_dir=out / "vllm_base",
                device_map="cpu",
            )
        vllm_base = broadcast_object(vllm_base)
    cell = "rl" if method == "rl" else "rl_opd"
    lambda_opd = cell_lambda(cell, float(getattr(args, "lambda_opd", 0.0) or 0.0))
    teacher_fn = None if lambda_opd <= 0 else teacher_for(
        args.component,
        harness=getattr(args, "harness", None),
        teacher_kind=str(getattr(args, "teacher_kind", "upstream") or "upstream"),
    )
    collection_mode = collection_mode_for_cell(cell, lambda_opd)
    loop = HybridLoopState(policy_version="v0")
    sampler = QuerySampler(
        train_rows,
        base_seed=int(args.seed),
        groups_per_step=int(getattr(args, "train_groups_per_step", 32) or 32),
    )
    adapter_live: str | None = None
    last_ckpt: str | None = None
    if getattr(args, "resume", False) and is_coordinator():
        resume_state = load_training_resume(out / "checkpoints" / cell)
        if resume_state is None:
            raise SystemExit(f"--resume set but no checkpoint under {out / 'checkpoints' / cell}")
        sampler = QuerySampler(
            train_rows,
            base_seed=int(args.seed),
            groups_per_step=int(getattr(args, "train_groups_per_step", 32) or 32),
            state=QuerySamplerState.from_dict(resume_state["sampler_state"]),
        )
        loop.policy_version = str(resume_state["updated_policy_version"] or "v0")
        adapter_live = str(resume_state["adapter_dir"])
        last_ckpt = str(resume_state["checkpoint_dir"])
        if resume_state.get("rng_path"):
            load_rng_state(Path(resume_state["rng_path"]))
    sampler_state = broadcast_object(sampler.state.to_dict() if is_coordinator() else None)
    if not is_coordinator():
        sampler = QuerySampler(
            train_rows,
            base_seed=int(args.seed),
            groups_per_step=int(getattr(args, "train_groups_per_step", 32) or 32),
            state=QuerySamplerState.from_dict(sampler_state),
        )
        loop.policy_version = str(broadcast_object(None))
        adapter_live = broadcast_object(None)
        last_ckpt = broadcast_object(None)
    else:
        broadcast_object(loop.policy_version)
        broadcast_object(adapter_live)
        last_ckpt = broadcast_object(last_ckpt)

    resolved = {
        "training_backend": "verl_fsdp2",
        "train_method": method,
        "n_gpus": int(dist.world_size),
        "tensor_parallel_size": 1,
        "rollout_replicas": int(dist.world_size),
        "layout": f"TP1 x {int(dist.world_size)}",
        "cispo": resolved_cispo_config(),
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "max_turns": int(args.max_turns),
        "max_model_len": int(getattr(args, "max_model_len", 8192) or 8192),
        "train_steps": int(args.train_steps),
        "train_groups_per_step": int(getattr(args, "train_groups_per_step", 32) or 32),
        "group_size": int(args.group_size),
        "enforce_eager": bool(getattr(args, "enforce_eager", False)),
        "collection_mode": collection_mode,
        "pool": pool_meta,
    }
    _write_resolved_config(out, resolved)
    if is_coordinator():
        print(json.dumps({"event": "verl_fsdp2_start", **resolved}, indent=2), flush=True)

    target = int(args.train_steps)
    max_empty = int(getattr(args, "max_empty_rollouts", 8) or 8)
    empty_streak = 0
    metrics_path = out / cell / "metrics.jsonl"
    session_root = out / "vllm_sessions"
    shard_dir = out / "tmp" / "rollout_shards"
    from trim.integrations.verl.fsdp2_actor import FSDP2CispoActor

    actor: FSDP2CispoActor | None = None
    already = int(sampler.state.global_optimizer_step)
    while already < target:
        if empty_streak >= max_empty:
            raise RuntimeError(f"too many empty-signal rollouts ({empty_streak}) before {target} updates")
        if is_coordinator():
            step_rows, sample_meta = sampler.sample_for_rollout()
            sampler.note_rollout_start()
        else:
            step_rows, sample_meta = [], {}
        packed = broadcast_object(
            {
                "rows": step_rows if is_coordinator() else None,
                "meta": sample_meta if is_coordinator() else None,
                "sampler": sampler.state.to_dict() if is_coordinator() else None,
            }
        )
        step_rows = list(packed["rows"])
        sample_meta = dict(packed["meta"])
        sampler.state = QuerySamplerState.from_dict(packed["sampler"])
        rollout_batch_id = int(sampler.state.global_rollout_batch)
        local_rows = shard_for_rank(step_rows, rank=int(dist.rank), world_size=int(dist.world_size))
        wait_gpus_quiet()
        if actor is not None:
            actor.close()
            actor = None
            wait_gpus_quiet()
        tag = f"{cell}_b{rollout_batch_id}_r{int(dist.rank)}"
        session_dir = session_root / tag
        session_dir.mkdir(parents=True, exist_ok=True)
        client = VLLMGenerateClient(
            model_path=vllm_base,
            session_dir=session_dir,
            tensor_parallel_size=1,
            max_model_len=int(getattr(args, "max_model_len", 8192) or 8192),
            lora_path=(
                adapter_live
                if adapter_live and (Path(adapter_live) / "adapter_model.safetensors").is_file()
                else None
            ),
            gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.90) or 0.90),
            enforce_eager=bool(getattr(args, "enforce_eager", False)),
            max_num_seqs=max(8, int(getattr(args, "max_num_seqs", 256) or 256) // max(1, int(dist.world_size))),
        )
        print(
            f"[verl-fsdp2 rank{dist.rank}] rollout batch={rollout_batch_id} "
            f"opt={already}/{target} local_queries={len(local_rows)} eager={client.enforce_eager}",
            flush=True,
        )
        t_roll = time.perf_counter()
        try:
            client.start()
            groups = rollout_queries_batched(
                client.generate_batch,
                local_rows,
                component_id=args.component,
                group_size=int(args.group_size),
                max_turns=int(args.max_turns),
                max_new=int(args.max_new_tokens),
                policy_version=loop.policy_version,
                seed=int(args.seed) + rollout_batch_id,
                sample=True,
                enc=enc,
                searcher=train_searcher,
                harness_mask=resolved_rollout_mask(args.component, harness=getattr(args, "harness", None)),
                train_env=str(getattr(args, "train_env", "local_legacy") or "local_legacy"),
                collection_mode=collection_mode,
            )
        finally:
            client.close()
            wait_gpus_quiet()
        rollout_s = time.perf_counter() - t_roll
        all_groups = all_gather_via_disk(groups, shard_dir=shard_dir, tag=f"b{rollout_batch_id}")
        rl_groups, n_const = drop_constant_reward_groups(all_groups)
        opd_datums: list[Any] = []
        if lambda_opd > 0 and teacher_fn is not None:
            if is_coordinator():
                batch = prepare_hybrid_batch(
                    groups=all_groups,
                    rl_datums_by_query={
                        g.query_id: list((g.trajectory_group or {}).get("rl_rows") or []) for g in all_groups
                    },
                    policy_version=loop.policy_version,
                    lambda_opd=lambda_opd,
                    component_id=args.component,
                    teacher_event_fn=teacher_fn,
                    encode_fn=enc.encode,
                    model_enc=enc,
                    opd_states_per_trajectory=int(getattr(args, "opd_states_per_trajectory", 3) or 3),
                    remove_constant_reward_groups=False,
                    include_format_errors=False,
                    seed=int(args.seed) + rollout_batch_id,
                    opd_loss=str(getattr(args, "opd_loss", "sr_opd_ce") or "sr_opd_ce"),
                    opd_gate_beta=float(getattr(args, "opd_gate_beta", 5.0) or 5.0),
                )
                opd_datums = list(batch.opd_datums)
            opd_datums = broadcast_object(opd_datums)
        rl_rows = training_rows_from_groups(rl_groups)
        if not rl_rows and not opd_datums:
            empty_streak += 1
            if is_coordinator():
                print(f"[verl-fsdp2] skip empty batch={rollout_batch_id} const_groups={n_const}", flush=True)
            continue
        local_train = shard_training_rows(rl_rows, rank=int(dist.rank), world_size=int(dist.world_size))
        local_opd = shard_for_rank(list(opd_datums), rank=int(dist.rank), world_size=int(dist.world_size)) if opd_datums else []
        opt_path = None
        if last_ckpt:
            cand = Path(last_ckpt) / f"optimizer.rank{int(dist.rank):04d}.pt"
            if cand.is_file():
                opt_path = str(cand)
        actor = FSDP2CispoActor(
            model_path=str(args.base_model or args.model_name),
            adapter_dir=adapter_live,
            learning_rate=1e-5,
            micro_batch_size=int(getattr(args, "train_micro_batch_size", 4) or 4),
            max_full_tokens=int(getattr(args, "max_model_len", 8192) or 8192),
            optimizer_path=opt_path,
        )
        t_train = time.perf_counter()
        part = actor.update(local_train, local_opd, lambda_opd=lambda_opd)
        train_s = time.perf_counter() - t_train
        n_opt = int(part.get("n_optimizer_steps") or 0)
        if n_opt > 0:
            sampler.note_update_complete()
            loop.bump_after_update()
            empty_streak = 0
            already = int(sampler.state.global_optimizer_step)
            ckpt_tmp = out / "checkpoints" / cell / f".tmp_step_{already:06d}"
            ckpt_final = out / "checkpoints" / cell / f"step_{already:06d}"
            adapter_step = ckpt_tmp / "adapter"
            ckpt_tmp.mkdir(parents=True, exist_ok=True)
            actor.save_adapter(adapter_step)
            import torch

            torch.save(actor.optimizer.state_dict(), ckpt_tmp / f"optimizer.rank{int(dist.rank):04d}.pt")
            barrier()
            if is_coordinator():
                save_rng_state(ckpt_tmp / "rng.json")
                (ckpt_tmp / "sampler.json").write_text(
                    json.dumps(sampler.state.to_dict(), indent=2) + "\n", encoding="utf-8"
                )
                audit = reward_parts_group_stats(all_groups)
                manifest = {
                    "cell": cell,
                    "step": already,
                    "updated_policy_version": loop.policy_version,
                    "n_optimizer_steps": n_opt,
                    "rollout_batch_id": rollout_batch_id,
                    "sample_meta": sample_meta,
                    "train": part,
                    "rollout_s": round(rollout_s, 3),
                    "train_s": round(train_s, 3),
                    "n_const_reward_groups": n_const,
                    "reward_group_audit": audit,
                    "layout": resolved["layout"],
                }
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "step": already,
                        "n_optimizer_steps": n_opt,
                        "n_rl_datums": len(rl_rows),
                        "n_opd_datums": len(opd_datums),
                        "rl_loss_proxy": part.get("loss"),
                        "rollout_s": round(rollout_s, 3),
                        "train_s": round(train_s, 3),
                        "successful_optimizer_step": already,
                        "policy_version": loop.policy_version,
                    }) + "\n")
                publish_step_checkpoint(ckpt_tmp, ckpt_final, manifest=manifest)
                live = out / "adapters" / cell
                live.mkdir(parents=True, exist_ok=True)
                import shutil

                src_adapter = ckpt_final / "adapter"
                if src_adapter.is_dir():
                    shutil.copytree(src_adapter, live, dirs_exist_ok=True)
                adapter_live = str(live)
                last_ckpt = str(ckpt_final)
            adapter_live = broadcast_object(adapter_live if is_coordinator() else None)
            last_ckpt = broadcast_object(last_ckpt if is_coordinator() else None)
            loop.policy_version = str(broadcast_object(loop.policy_version if is_coordinator() else None))
            sampler.state = QuerySamplerState.from_dict(
                broadcast_object(sampler.state.to_dict() if is_coordinator() else None)
            )
            already = int(sampler.state.global_optimizer_step)
        else:
            empty_streak += 1
        actor.close()
        actor = None
        wait_gpus_quiet()
        barrier()

    if is_coordinator():
        print(json.dumps({"ok": True, "successful_optimizer_steps": already, "backend": "verl_fsdp2"}), flush=True)
    return {
        "ok": True,
        "training_backend": "verl_fsdp2",
        "n_optimizer_steps": already,
        "train_only": True,
        "component": getattr(args, "component", None),
    }
