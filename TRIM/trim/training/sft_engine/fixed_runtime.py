"""Offline Teacher and Projected Teacher loop. Every CE rank steps together.

Model creation, checkpointing and adapter export are dispatched through
``sft_engine.backend`` so ``accelerate_ddp`` keeps its historical behaviour and
``accelerate_fsdp2`` shards the frozen base. The batch loop (slot plan, manual
global mass, one optimizer step) is shared and unchanged.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.contracts import fingerprint, package_digest
from trim.training.sft_ablation.fixed_train import select_batch_indices
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_tree
from trim.training.sft_ablation.identity import tokenizer_digest
from trim.training.sft_ablation.session import append_metrics
from trim.training.sft_engine.backend import backend_name
from trim.training.sft_engine.checkpoint import latest_v2
from trim.training.sft_engine.contract import adapter_config_from, build_v2_contract, require_exact_resume
from trim.training.sft_engine.data import load_sealed
from trim.training.sft_engine.distributed import all_gather_objects, broadcast_ints
from trim.training.sft_engine.profile import RankProgress, summarize_progress


def run_fixed_sft(args: Any) -> int:
    experiment = str(args.experiment)
    if experiment not in {"teacher_sft", "projected_teacher_sft"}:
        raise SystemExit(f"fixed SFT engine does not run {experiment}")
    if str(getattr(args, "stage", "")) != "train":
        raise SystemExit("fixed SFT engine only accepts --stage train")
    if not str(getattr(args, "sft_data", "") or "") or not str(getattr(args, "dataset_manifest", "") or ""):
        raise SystemExit("train-only requires --sft-data and --dataset-manifest")
    backend = backend_name(args)
    if int(args.micro_batch_size) != 1:
        raise SystemExit(f"{backend} requires --micro-batch-size 1")

    from trim.training.sft_engine.backend import make_training_accelerator
    from trim.training.sft_engine.model import (
        assert_distinct_gpu_uuids,
        device_uuid,
        placement_record,
    )

    dataset = load_sealed(
        manifest_path=Path(args.dataset_manifest),
        rows_path=Path(args.sft_data),
        expected_experiment=experiment,
    )
    live_tokenizer = tokenizer_digest(str(args.model_name))
    if live_tokenizer != dataset.tokenizer_digest:
        raise SystemExit(
            "tokenizer digest does not match the sealed dataset "
            f"({live_tokenizer} != {dataset.tokenizer_digest})"
        )
    population = [
        int(item["position"])
        for item in dataset.manifest["row_index"]
        if item.get("train", True) and int(item.get("length", 0)) <= int(args.max_model_len)
    ]
    if not population:
        raise SystemExit("sealed dataset has no rows within max_model_len")
    train_rows = [item for item in dataset.manifest["row_index"] if item.get("train", True)]
    too_long = sum(1 for item in train_rows if int(item.get("length", 0)) > int(args.max_model_len))
    print(
        f"[fixed_sft] backend={backend} training population {len(population)} rows "
        f"({too_long} train rows exceed max_model_len={int(args.max_model_len)} and are excluded)",
        flush=True,
    )

    accelerator = make_training_accelerator(args)
    progress = RankProgress(
        Path(args.out),
        rank=int(accelerator.process_index),
        gpu_uuid=device_uuid(accelerator.device),
    )
    try:
        return _train(
            args,
            experiment=experiment,
            dataset=dataset,
            population=population,
            accelerator=accelerator,
            progress=progress,
            placement_record=placement_record,
            assert_distinct_gpu_uuids=assert_distinct_gpu_uuids,
        )
    except Exception as exc:
        progress.event("failed", error=f"{type(exc).__name__}: {exc}")
        if int(accelerator.process_index) == 0:
            atomic_write_json(
                Path(args.out) / "FAILED.json",
                {"rank": int(accelerator.process_index), "error": f"{type(exc).__name__}: {exc}"},
            )
        raise
    finally:
        progress.close()


def _train(
    args: Any,
    *,
    experiment: str,
    dataset: Any,
    population: list[int],
    accelerator: Any,
    progress: RankProgress,
    placement_record: Any,
    assert_distinct_gpu_uuids: Any,
) -> int:
    from trim.training.sft_engine.backend import create_training_context, export_policy_adapter, save_training_checkpoint, update

    out = Path(args.out)
    if int(accelerator.process_index) == 0:
        out.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    placement = placement_record(accelerator)
    records = all_gather_objects(placement)
    assert_distinct_gpu_uuids(accelerator, records)
    progress.event("placement", **placement, peers=records if accelerator.process_index == 0 else [])

    resume_dir = Path(args.resume) if str(getattr(args, "resume", "") or "") else latest_v2(out / "checkpoints")
    adapter_path = str(resume_dir / "adapter") if resume_dir is not None else str(args.init_adapter)
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"missing adapter: {adapter_path or args.init_adapter}")
    init_digest = sha256_tree(Path(args.init_adapter)) if str(args.init_adapter or "") else "missing"
    backend = backend_name(args)
    contract = build_v2_contract(
        args,
        dataset_digest=dataset.package_digest,
        source_contract_digest=str(dataset.manifest["source_contract_sha256"]),
        tokenizer_digest=dataset.tokenizer_digest,
        gradient_ranks=int(accelerator.num_processes),
        nnodes=int(os.environ.get("NNODES") or _inferred_nnodes(accelerator)),
        nproc_per_node=int(os.environ.get("LOCAL_WORLD_SIZE") or accelerator.num_processes),
        init_adapter_digest=init_digest,
        backend=backend,
        adapter_config=adapter_config_from(adapter_path),
    )
    _seal_contract(out, contract, accelerator)

    ctx = create_training_context(
        args,
        adapter_path=adapter_path,
        contract=contract,
        resume_dir=resume_dir,
        progress=lambda phase, **fields: progress.event(phase, **fields),
        accelerator=accelerator,
    )
    model = ctx.model
    step = 0
    cursor = 0
    if resume_dir is not None:
        loaded = ctx.resume_manifest
        if loaded is None:
            raise RuntimeError("resume directory was not restored")
        step = int(loaded["optimizer_step"])
        cursor = int(loaded["sampler"]["cursor"])
    progress.event("ready", global_step=step, cursor=cursor, n_train_rows=len(population))

    target = int(args.num_updates)
    while step < target:
        indices = None
        if int(accelerator.process_index) == 0:
            logical, cursor, _uses = select_batch_indices(
                len(population),
                cursor=cursor,
                batch_size=int(args.global_batch_size),
                seed=int(args.seed),
            )
            indices = [population[item] for item in logical]
        indices = broadcast_ints(indices, device=accelerator.device, src=0)
        rows = [dataset.get(index) for index in indices]
        for row in rows:
            if row.experiment != experiment:
                raise RuntimeError(f"global batch contains {row.experiment}")
        started = _now()
        result = update(
            ctx,
            rows,
            int(args.max_model_len),
            grad_clip=float(args.grad_clip),
            trace=lambda phase, **fields: progress.event(phase, global_step=step + 1, **fields),
            sync_cuda=step < int(getattr(args, "profile_steps", 0) or 0),
        )
        if not result["updated"]:
            raise RuntimeError("nonempty SFT batch produced no update")
        step += 1
        wall = _now() - started
        source_digest = package_digest(rows)
        selection = fingerprint([row.row_id for row in rows])
        saved_digest = ""
        if _should_save(step, int(args.save_every), target):
            published = save_training_checkpoint(
                ctx,
                out / "checkpoints" / f"step-{step:06d}",
                contract=contract,
                optimizer_step=step,
                policy_version=f"{experiment}:step{step}",
                source_digest=source_digest,
                rollout_adapter_digest="",
                sampler={"cursor": cursor, "row_ids": [row.row_id for row in rows]},
                selection_digest=selection,
            )
            if int(accelerator.process_index) == 0:
                saved_digest = str(read_json(published / "SFT_CHECKPOINT.json")["adapter_digest"])
        progress.event(
            "optimizer_step",
            global_step=step,
            update_wall_s=wall,
            real_rows=result["real_rows"],
            dummy_rows=result["dummy_rows"],
            input_tokens=result["input_tokens"],
            supervised_tokens=result["supervised_tokens"],
            weighted_tokens=result["weighted_tokens"],
            forward_s=result["forward_s"],
            backward_s=result["backward_s"],
            sync_s=result["sync_s"],
            optimizer_s=result["optimizer_s"],
            loss_ce=result["loss_ce"],
            grad_norm=result["grad_norm"],
            memory_allocated=_memory(accelerator)[0],
            memory_reserved=_memory(accelerator)[1],
            peak_allocated=_memory(accelerator)[2],
        )
        if int(accelerator.process_index) == 0:
            append_metrics(
                out,
                {
                    "experiment_id": experiment,
                    "optimizer_step": step,
                    "policy_version": f"{experiment}:step{step}",
                    "adapter_digest": saved_digest,
                    "source_digest": source_digest,
                    "selection_digest": selection,
                    "n_rows": result["n_rows"],
                    "weighted_tokens": result["weighted_tokens"],
                    "loss_ce": result["loss_ce"],
                    "grad_norm": result["grad_norm"],
                    "learning_rate": float(args.learning_rate),
                    "gradient_ranks": result["gradient_ranks"],
                    "update_wall_s": wall,
                    "input_tokens": result["input_tokens"],
                    "supervised_tokens": result["supervised_tokens"],
                    "dummy_rows": result["dummy_rows"],
                },
            )
    final = out / "adapters" / "final"
    export_policy_adapter(ctx, final)
    if int(accelerator.process_index) == 0:
        atomic_write_json(out / "PERF_REPORT.json", summarize_progress(out))
    accelerator.wait_for_everyone()
    return 0


def _seal_contract(out: Path, contract: dict[str, Any], accelerator: Any) -> None:
    path = out / "CONTRACT_V2.json"
    if int(accelerator.process_index) == 0 and not path.is_file():
        atomic_write_json(path, contract)
    accelerator.wait_for_everyone()
    saved = read_json(path)
    require_exact_resume(saved, contract)


def _should_save(step: int, save_every: int, target: int) -> bool:
    if step == 1 or step == target:
        return True
    return save_every > 0 and step % save_every == 0


def _seed_rank(seed: int, rank: int) -> None:
    import torch

    random.seed(int(seed) + int(rank) * 100003)
    torch.manual_seed(int(seed) + int(rank) * 100003)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + int(rank) * 100003)


def _inferred_nnodes(accelerator: Any) -> int:
    local = int(os.environ.get("LOCAL_WORLD_SIZE") or accelerator.num_processes or 1)
    return max(1, int(accelerator.num_processes) // max(1, local))


def _now() -> float:
    import time

    return time.perf_counter()


def _memory(accelerator: Any) -> tuple[int, int, int]:
    import torch

    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0, 0
    index = accelerator.device.index
    return (
        int(torch.cuda.memory_allocated(index)),
        int(torch.cuda.memory_reserved(index)),
        int(torch.cuda.max_memory_allocated(index)),
    )
