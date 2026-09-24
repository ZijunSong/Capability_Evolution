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
    plan_joint_sync_batches,
    training_rows_from_groups,
)
from trim.integrations.verl.joint_objective import resolved_cispo_config
from trim.training.dist_runtime import (
    TRIM_EXPECTED_WORLD_SIZE_ENV,
    TRIM_OUTER_RANK_ENV_JSON,
    TRIM_TRAIN_WORKER_ENV,
    all_gather_via_disk,
    resolve_rollout_shard_dir,
    assert_expected_world_size,
    assert_unique_local_gpu_uuids,
    barrier,
    broadcast_object,
    collect_rank_probe,
    expected_world_size_from_env,
    init_dist_if_needed,
    is_coordinator,
    isolate_inference_child_env,
    pin_local_cuda_device,
    shard_for_rank,
    torchrun_env_source,
)
from trim.training.opd_train_contract import (
    assert_train_contract,
    opd_loss_from_args,
    training_cell_for_method,
)
from trim.training.rl_opd_types import TRAINING_MODE_RL, TRAINING_MODE_RL_OPD


VERL_METHODS = {"rl", "rl+opd", "trim", "scape_seed", "scape+seed"}


def require_verl_trainer() -> dict[str, Any]:
    """TRIM owns the actor loop. This is not a native verl engine import check."""
    return {
        "native_verl": False,
        "require_verl_trainer": "noop",
        "actual_engine": "trim_fsdp2_or_ddp_lora",
    }


def _requested_world_size(args: Any, dist_world: int) -> int:
    raw = getattr(args, "expected_world_size", None)
    if raw in {None, 0, ""}:
        raw = expected_world_size_from_env()
    if raw in {None, 0, ""}:
        return max(1, int(dist_world))
    return max(1, int(raw))


def resolve_resume_optimizer_path(
    ckpt_dir: str | Path,
    *,
    rank: int,
    world_size: int,
    wrap: str,
) -> str:
    ckpt = Path(ckpt_dir)
    if str(wrap) == "ddp":
        for name in ("optimizer.pt", "optimizer.rank0000.pt"):
            cand = ckpt / name
            if cand.is_file():
                return str(cand)
        raise SystemExit(
            f"DDP resume requires a full optimizer state under {ckpt} "
            "(optimizer.pt or optimizer.rank0000.pt). Missing state must not reset Adam."
        )
    cand = ckpt / f"optimizer.rank{int(rank):04d}.pt"
    if cand.is_file():
        return str(cand)
    single = ckpt / "optimizer.rank0000.pt"
    if int(world_size) > 1 and single.is_file():
        raise SystemExit(
            f"FSDP2 cannot restore world_size={world_size} from single-rank {single}. "
            "Use --training-backend torch_ddp_lora to load the same full optimizer on every rank, "
            "or start a new --out and keep the old checkpoint."
        )
    raise SystemExit(f"missing optimizer state {cand}")


def _gather_probes(local: dict[str, Any]) -> list[dict[str, Any]]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return [local]
    payload: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(payload, local)
    return [dict(row or {}) for row in payload]


def _broadcast_run_id(out: Path) -> str:
    raw = os.environ.get("TRIM_RUN_ID") or os.environ.get("TORCHELASTIC_RUN_ID") or ""
    if is_coordinator() and not str(raw).strip():
        raw = f"{out.name}-{int(time.time())}-{os.getpid()}"
    run_id = broadcast_object(str(raw) if is_coordinator() else None)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(run_id or "run"))
    return safe or "run"


def _rollout_component_seconds(groups: list[Any]) -> dict[str, float]:
    keys = ("model_sec", "prompt_sec", "snapshot_sec", "parse_sec", "execute_sec", "freeze_sec", "harness_sec")
    acc = {f"rollout_component_{key}": 0.0 for key in keys}
    for group in groups:
        trajectory = getattr(group, "trajectory_group", None) or {}
        if not isinstance(trajectory, dict):
            continue
        for episode in trajectory.get("episode_stats") or []:
            if not isinstance(episode, dict):
                continue
            for key in keys:
                acc[f"rollout_component_{key}"] += float(episode.get(key) or 0.0)
    return acc


def _emit_step_timing(
    marks: dict[str, Any],
    *,
    timing_path: Path,
    step: int | None,
    rollout_batch_id: int,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, float]:
    from trim.training.runtime_manifest import finalize_step_timing, summarize_rank_timings

    local = finalize_step_timing(marks)
    rows = _gather_probes(local)
    if is_coordinator():
        record = {
            "step": step,
            "rollout_batch_id": int(rollout_batch_id),
            "status": status,
            "ranks": rows,
            "summary": summarize_rank_timings(rows),
        }
        if extra:
            record["extra"] = extra
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    return local


def _actor_wrap(backend: str) -> str:
    return "ddp" if str(backend).lower().replace("-", "_") == "torch_ddp_lora" else "fsdp2"


def _unsupported_method(method: str, opd_loss: str | None = None, backend: str | None = None) -> None:
    assert_train_contract(method, opd_loss, backend or "verl")


def _write_resolved_config(out: Path, payload: dict[str, Any]) -> None:
    if is_coordinator():
        (out / "RESOLVED_CONFIG.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_verl_fsdp2_train(args: Any) -> dict[str, Any]:
    """On-policy loop used by ``run_train.py --training-backend verl``."""
    os.environ.setdefault("TRIM_DIST_BACKEND", "nccl")
    os.environ["TRIM_GPU_KEEPALIVE"] = "0"
    pin_local_cuda_device()
    dist = init_dist_if_needed(backend="nccl")
    backend = str(getattr(args, "training_backend", "verl") or "verl").lower().replace("-", "_")
    wrap = _actor_wrap(backend)
    requested = _requested_world_size(args, dist.world_size)
    assert_expected_world_size(dist, requested)
    probes = _gather_probes(collect_rank_probe())
    if is_coordinator():
        assert_unique_local_gpu_uuids(probes)
        print(
            json.dumps(
                {
                    "event": "topology_confirmed",
                    "requested_world_size": requested,
                    "effective_world_size": int(dist.world_size),
                    "wrap": wrap,
                    "native_verl": False,
                    "probes": probes,
                    "source": torchrun_env_source(),
                    "outer_rank_env": os.environ.get(TRIM_OUTER_RANK_ENV_JSON),
                    "trim_train_worker": os.environ.get(TRIM_TRAIN_WORKER_ENV),
                    "require_verl_trainer": require_verl_trainer(),
                },
                indent=2,
            ),
            flush=True,
        )
    barrier()
    _unsupported_method(
        str(getattr(args, "train_method", None) or getattr(args, "training_mode", "")),
        opd_loss_from_args(args),
        str(getattr(args, "training_backend", "verl") or "verl"),
    )

    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.batched_env_rollout import rollout_queries_batched, turn_audit_from_groups
    from trim.training.opd_batch_health import assess_opd_health
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
    opd_loss = opd_loss_from_args(args, method=method)
    _unsupported_method(
        method,
        opd_loss,
        str(getattr(args, "training_backend", "verl") or "verl"),
    )

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

    train_rows, eval_rows, pool_meta, _frozen_points = resolve_queries(args)
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
    cell = training_cell_for_method(method)
    lambda_opd = cell_lambda(cell, float(getattr(args, "lambda_opd", 0.0) or 0.0))
    teacher_fn = None if lambda_opd <= 0 else teacher_for(
        args.component,
        harness=getattr(args, "harness", None),
        teacher_kind=str(getattr(args, "teacher_kind", "upstream") or "upstream"),
    )
    collection_mode = collection_mode_for_cell(cell, lambda_opd, opd_loss)
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
        "training_backend": backend,
        "requested": {
            "world_size": requested,
            "tensor_parallel_size": 1,
            "rollout_replicas": requested,
            "layout": f"TP1 x {requested}",
        },
        "effective": {
            "world_size": int(dist.world_size),
            "n_gpus": int(dist.world_size),
            "tensor_parallel_size": 1,
            "rollout_replicas": int(dist.world_size),
            "layout": f"TP1 x {int(dist.world_size)}",
            "wrap": wrap,
            "actor_class": "DDPLoraActor" if wrap == "ddp" else "FSDP2CispoActor",
        },
        "engine": {
            "native_verl": False,
            "require_verl_trainer": "noop",
            "actual_actor_class": "DDPLoraActor" if wrap == "ddp" else "FSDP2CispoActor",
            "claimed_backend": backend,
        },
        "train_method": method,
        "cell": cell,
        "opd_loss": opd_loss,
        "lambda_opd": float(lambda_opd),
        "actor_wrap": wrap,
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
        "probes": probes,
    }
    _write_resolved_config(out, resolved)
    if is_coordinator():
        launch = {
            "event": "topology_launch",
            "out": str(out),
            "training_backend": backend,
            "requested_world_size": requested,
            "effective_world_size": int(dist.world_size),
            "layout": resolved["layout"],
            "engine": resolved["engine"],
            "dist": {
                "rank": int(dist.rank),
                "world_size": int(dist.world_size),
                "local_rank": int(dist.local_rank),
                "local_world_size": int(dist.local_world_size),
                "node_rank": int(dist.node_rank),
                "initialized": bool(dist.initialized),
            },
            "probes": probes,
        }
        (out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "verl_fsdp2_start", **resolved}, indent=2), flush=True)

    target = int(args.train_steps)
    max_empty = int(getattr(args, "max_empty_rollouts", 8) or 8)
    max_empty_opd = int(getattr(args, "max_empty_opd_batches", 3) or 3)
    empty_streak = 0
    opd_empty_streak = 0
    metrics_path = out / cell / "metrics.jsonl"
    timing_path = out / cell / "step_timing.jsonl"
    sample_path = out / cell / "turn_samples.jsonl"
    session_root = out / "vllm_sessions"
    run_id = _broadcast_run_id(out)
    local_world = int(getattr(dist, "local_world_size", 0) or 0)
    single_node = local_world > 0 and int(dist.world_size) == local_world
    shard_dir = resolve_rollout_shard_dir(out=out, run_id=run_id, single_node=single_node)
    if is_coordinator():
        resolved["run_id"] = run_id
        resolved["rollout_shard_dir"] = str(shard_dir)
        resolved["single_node_ipc"] = single_node
        _write_resolved_config(out, resolved)
        print(
            json.dumps(
                {
                    "event": "rollout_shard_dir",
                    "run_id": run_id,
                    "path": str(shard_dir),
                    "single_node": single_node,
                }
            ),
            flush=True,
        )
    from trim.integrations.verl.fsdp2_actor import DDPLoraActor, FSDP2CispoActor

    actor: FSDP2CispoActor | DDPLoraActor | None = None
    already = int(sampler.state.global_optimizer_step)
    while already < target:
        if empty_streak >= max_empty:
            raise RuntimeError(f"too many empty-signal rollouts ({empty_streak}) before {target} updates")
        t_step = time.perf_counter()
        marks: dict[str, Any] = {}
        if is_coordinator():
            step_rows, sample_meta = sampler.sample_for_rollout()
            sampler.note_rollout_start()
        else:
            step_rows, sample_meta = [], {}
        t_sample = time.perf_counter()
        packed = broadcast_object(
            {
                "rows": step_rows if is_coordinator() else None,
                "meta": sample_meta if is_coordinator() else None,
                "sampler": sampler.state.to_dict() if is_coordinator() else None,
            }
        )
        marks["sample_broadcast_s"] = time.perf_counter() - t_sample
        step_rows = list(packed["rows"])
        sample_meta = dict(packed["meta"])
        sampler.state = QuerySamplerState.from_dict(packed["sampler"])
        rollout_batch_id = int(sampler.state.global_rollout_batch)
        local_rows = shard_for_rank(step_rows, rank=int(dist.rank), world_size=int(dist.world_size))
        t_switch = time.perf_counter()
        wait_gpus_quiet()
        if actor is not None:
            actor.close()
            actor = None
            wait_gpus_quiet()
        marks["phase_switch_s"] = time.perf_counter() - t_switch
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
            extra_env=isolate_inference_child_env(),
        )
        print(
            f"[verl-fsdp2 rank{dist.rank}] rollout batch={rollout_batch_id} "
            f"opt={already}/{target} local_queries={len(local_rows)} eager={client.enforce_eager}",
            flush=True,
        )
        rollout_s = 0.0
        groups: list[Any] = []
        try:
            t_engine = time.perf_counter()
            client.start()
            marks["rollout_engine_start_s"] = time.perf_counter() - t_engine
            t_gen = time.perf_counter()
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
                opd_loss=opd_loss,
            )
            marks["rollout_generate_s"] = time.perf_counter() - t_gen
            rollout_s = float(marks["rollout_engine_start_s"]) + float(marks["rollout_generate_s"])
        finally:
            t_close = time.perf_counter()
            client.close()
            wait_gpus_quiet()
            marks["rollout_engine_close_s"] = time.perf_counter() - t_close
        marks.update(_rollout_component_seconds(groups))
        t_gather = time.perf_counter()
        all_groups = all_gather_via_disk(
            groups,
            shard_dir=shard_dir,
            tag=f"b{rollout_batch_id}_pv{loop.policy_version}",
            policy_version=str(loop.policy_version),
        )
        marks["gather_s"] = time.perf_counter() - t_gather
        rl_groups, n_const = drop_constant_reward_groups(all_groups)
        opd_datums: list[Any] = []
        projection_stats: dict[str, Any] = {}
        turn_audit = turn_audit_from_groups(all_groups)
        if lambda_opd > 0 and teacher_fn is not None:
            t_build = time.perf_counter()
            payload = None
            build_err = None
            if is_coordinator():
                try:
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
                    opd_states_per_trajectory=(
                        int(args.opd_states_per_trajectory)
                        if getattr(args, "opd_states_per_trajectory", None) is not None
                        else (-1 if cell == "scape_seed" else 3)
                    ),
                    remove_constant_reward_groups=False,
                    include_format_errors=False,
                    seed=int(args.seed) + rollout_batch_id,
                    opd_loss=opd_loss,
                    opd_gate_beta=float(getattr(args, "opd_gate_beta", 5.0) or 5.0),
                    )
                    payload = {
                        "opd_datums": list(batch.opd_datums),
                        "projection_stats": dict(batch.projection_stats),
                    }
                except Exception as exc:
                    build_err = f"{type(exc).__name__}: {exc}"
            blob = broadcast_object({"err": build_err, "payload": payload} if is_coordinator() else None)
            marks["teacher_projector_build_s"] = time.perf_counter() - t_build
            if blob and blob.get("err"):
                marks["_wall"] = time.perf_counter() - t_step
                _emit_step_timing(
                    marks,
                    timing_path=timing_path,
                    step=already,
                    rollout_batch_id=rollout_batch_id,
                    status="fatal_opd_build",
                    extra={"message": blob["err"], "turn_audit": turn_audit.get("counts")},
                )
                raise RuntimeError(str(blob["err"]))
            got = (blob or {}).get("payload") or {}
            opd_datums = list(got.get("opd_datums") or [])
            projection_stats = dict(got.get("projection_stats") or {})
        rl_rows = training_rows_from_groups(rl_groups)
        health = assess_opd_health(
            lambda_opd=float(lambda_opd),
            opd_loss=str(opd_loss),
            n_rl=len(rl_rows),
            n_opd=len(opd_datums),
            projection_stats=projection_stats,
            opd_empty_streak=opd_empty_streak,
            max_empty_opd=max_empty_opd,
        )
        opd_empty_streak = int(health["opd_empty_streak"])
        if health["fatal"]:
            marks["_wall"] = time.perf_counter() - t_step
            if is_coordinator() and turn_audit.get("samples"):
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                with sample_path.open("a", encoding="utf-8") as handle:
                    for sample in turn_audit["samples"]:
                        handle.write(
                            json.dumps({"step": already, "rollout_batch_id": rollout_batch_id, **sample}, default=str)
                            + "\n"
                        )
            _emit_step_timing(
                marks,
                timing_path=timing_path,
                step=already,
                rollout_batch_id=rollout_batch_id,
                status="fatal_opd",
                extra={
                    "message": health["message"],
                    "effective_update_type": health["effective_update_type"],
                    "projection": health["fields"],
                    "turn_audit": turn_audit.get("counts"),
                },
            )
            raise RuntimeError(health["message"])
        if not rl_rows and not opd_datums:
            empty_streak += 1
            marks["_wall"] = time.perf_counter() - t_step
            if is_coordinator():
                print(f"[verl-fsdp2] skip empty batch={rollout_batch_id} const_groups={n_const}", flush=True)
            _emit_step_timing(
                marks,
                timing_path=timing_path,
                step=already,
                rollout_batch_id=rollout_batch_id,
                status="skip_empty",
                extra={"effective_update_type": health["effective_update_type"], "turn_audit": turn_audit.get("counts")},
            )
            continue
        t_plan = time.perf_counter()
        plan = plan_joint_sync_batches(
            rl_rows,
            list(opd_datums),
            rank=int(dist.rank),
            world_size=int(dist.world_size),
            micro_batch_size=int(getattr(args, "train_micro_batch_size", 4) or 4),
        )
        marks["batch_plan_s"] = time.perf_counter() - t_plan
        opt_path = None
        if last_ckpt:
            opt_path = resolve_resume_optimizer_path(
                last_ckpt, rank=int(dist.rank), world_size=int(dist.world_size), wrap=wrap
            )
        actor_cls = DDPLoraActor if wrap == "ddp" else FSDP2CispoActor
        t_load = time.perf_counter()
        actor = actor_cls(
            model_path=str(args.base_model or args.model_name),
            adapter_dir=adapter_live,
            learning_rate=1e-5,
            micro_batch_size=int(getattr(args, "train_micro_batch_size", 4) or 4),
            max_full_tokens=int(getattr(args, "max_model_len", 8192) or 8192),
            optimizer_path=opt_path,
            wrap=wrap,
            heartbeat_every=int(getattr(args, "train_heartbeat_every", 8) or 8),
        )
        if is_coordinator() and getattr(actor, "runtime_info", None):
            (out / "RUNTIME_MANIFEST.json").write_text(
                json.dumps(
                    {
                        **dict(actor.runtime_info),
                        "rollout_batch_id": rollout_batch_id,
                        "optimizer_step_before": already,
                        "run_id": run_id,
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        barrier()
        marks["actor_load_s"] = time.perf_counter() - t_load
        t_train = time.perf_counter()
        part = actor.update(rl_rows, list(opd_datums), lambda_opd=lambda_opd, plan=plan)
        train_s = time.perf_counter() - t_train
        for key in ("teacher_score_s", "student_forward_s", "loss_s", "backward_s", "grad_sync_s", "optimizer_s"):
            marks[key] = float(part.get(key) or 0.0)
        inner = sum(float(marks[key]) for key in ("teacher_score_s", "student_forward_s", "loss_s", "backward_s", "grad_sync_s", "optimizer_s"))
        marks["actor_update_other_s"] = max(0.0, train_s - inner)
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
            t_adapter = time.perf_counter()
            actor.save_adapter(adapter_step)
            marks["adapter_save_s"] = time.perf_counter() - t_adapter
            import torch

            t_opt_save = time.perf_counter()
            state = actor.optimizer.state_dict()
            if wrap == "ddp":
                if is_coordinator():
                    torch.save(state, ckpt_tmp / "optimizer.pt")
            else:
                torch.save(state, ckpt_tmp / f"optimizer.rank{int(dist.rank):04d}.pt")
            del state
            barrier()
            marks["optimizer_save_s"] = time.perf_counter() - t_opt_save
            t_publish = time.perf_counter()
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
                    "wrap": wrap,
                    "train_method": method,
                    "opd_loss": opd_loss,
                    "lambda_opd": float(lambda_opd),
                    "requested_world_size": requested,
                    "effective_world_size": int(dist.world_size),
                    "batch_plan": {
                        "n_sync_rounds": plan.n_sync_rounds,
                        "n_dummy_rl": plan.n_dummy_rl,
                        "n_dummy_opd": plan.n_dummy_opd,
                        "n_global_rl_microbatches": plan.n_global_rl_microbatches,
                        "global_rl_tokens": plan.global_rl_tokens,
                        "global_opd_weight": plan.global_opd_weight,
                    },
                    "effective_update_type": health["effective_update_type"],
                    "projection": health["fields"],
                    "turn_audit": turn_audit.get("counts"),
                    "run_id": run_id,
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
                        "effective_update_type": health["effective_update_type"],
                        "opd_empty_streak": opd_empty_streak,
                        "Z_gap": part.get("Z_gap"),
                        "teacher_score_s": part.get("teacher_score_s"),
                        "projected_gap_raw": part.get("projected_gap_raw"),
                        "opd_objective": part.get("opd_objective"),
                        **health["fields"],
                        "turn_audit": turn_audit.get("counts"),
                    }, default=str) + "\n")
                if turn_audit.get("samples"):
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    with sample_path.open("a", encoding="utf-8") as handle:
                        for sample in turn_audit["samples"]:
                            handle.write(
                                json.dumps(
                                    {"step": already, "rollout_batch_id": rollout_batch_id, **sample},
                                    default=str,
                                )
                                + "\n"
                            )
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
            marks["adapter_publish_s"] = time.perf_counter() - t_publish
            already = int(sampler.state.global_optimizer_step)
            step_status = "updated"
        else:
            empty_streak += 1
            step_status = "no_optimizer_step"
        t_tail = time.perf_counter()
        actor.close()
        actor = None
        wait_gpus_quiet()
        t_barrier = time.perf_counter()
        marks["phase_switch_s"] = float(marks.get("phase_switch_s") or 0.0) + (t_barrier - t_tail)
        barrier()
        marks["barrier_wait_s"] = time.perf_counter() - t_barrier
        marks["_wall"] = time.perf_counter() - t_step
        _emit_step_timing(
            marks,
            timing_path=timing_path,
            step=already,
            rollout_batch_id=rollout_batch_id,
            status=step_status,
            extra={
                "effective_update_type": health["effective_update_type"],
                "n_rl_datums": len(rl_rows),
                "n_opd_datums": len(opd_datums),
                "turn_audit": turn_audit.get("counts"),
                "train_s": round(train_s, 3),
                "rollout_s": round(rollout_s, 3),
            },
        )

    if is_coordinator():
        print(json.dumps({"ok": True, "successful_optimizer_steps": already, "backend": "verl_fsdp2"}), flush=True)
    return {
        "ok": True,
        "training_backend": "verl_fsdp2",
        "n_optimizer_steps": already,
        "train_only": True,
        "component": getattr(args, "component", None),
    }
