"""Training backend dispatch for the three SFT experiments.

Five operations only — create, update, save, restore, export — so Teacher,
Projected Teacher and Student on-policy all pick the same backend semantics
without duplicating load/checkpoint logic. Data and environment logic stays in
the experiment runtimes.

``accelerate_ddp`` keeps the historical per-rank full-weight behaviour.
``accelerate_fsdp2`` shards the frozen base across gradient ranks (see
``model.load_fsdp2_model`` and ``sharded_checkpoint``). They do not intermix:
the run contract records the backend and exact resume refuses a backend change.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable

ACCELERATED_BACKENDS = ("accelerate_ddp", "accelerate_fsdp2")


def is_accelerated(backend: Any) -> bool:
    return str(backend or "legacy") in ACCELERATED_BACKENDS


def backend_name(args: Any) -> str:
    name = str(getattr(args, "ce_backend", "legacy") or "legacy")
    if name not in ACCELERATED_BACKENDS:
        raise ValueError(f"engine backend dispatch does not handle {name}")
    return name


def seed_rank(seed: int, rank: int) -> None:
    import torch

    random.seed(int(seed) + int(rank) * 100003)
    torch.manual_seed(int(seed) + int(rank) * 100003)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + int(rank) * 100003)


def make_training_accelerator(args: Any) -> Any:
    """Create the accelerator for the requested engine (one per process)."""
    name = backend_name(args)
    if name == "accelerate_fsdp2":
        from trim.training.sft_engine.model import make_fsdp2_accelerator

        return make_fsdp2_accelerator()
    from trim.training.sft_engine.model import make_accelerator

    return make_accelerator()


def create_training_context(
    args: Any,
    *,
    adapter_path: str,
    contract: dict[str, Any] | None = None,
    resume_dir: Path | None = None,
    progress: Callable[..., None] | None = None,
    accelerator: Any = None,
) -> Any:
    """Build accelerator + model + optimizer and restore any committed state.

    DDP keeps its staggered per-slot load. FSDP2 loads on every rank together
    (prepare itself is collective) and probes only after sharding.
    """
    from types import SimpleNamespace

    from trim.training.sft_ablation.io_util import read_json
    from trim.training.sft_engine.contract import require_exact_resume
    from trim.training.sft_engine.model import (
        build_optimizer,
        kernel_report,
        trainable_checksum,
    )
    from trim.training.sft_engine.distributed import all_gather_objects

    name = backend_name(args)
    attn_impl = str(getattr(args, "attn_implementation", "") or "") or None
    grad_ckpt = str(getattr(args, "gradient_checkpointing", "off") or "off")

    if name == "accelerate_fsdp2":
        from trim.training.sft_engine.model import (
            assert_fsdp2_ready,
            load_fsdp2_model,
            make_fsdp2_accelerator,
        )

        acc = accelerator if accelerator is not None else make_fsdp2_accelerator()
        if progress is not None:
            progress("load_model", backend=name)
        model = load_fsdp2_model(
            str(args.model_name),
            adapter_path,
            acc,
            attn_impl=attn_impl,
            gradient_checkpointing=grad_ckpt,
        )
        optimizer, scheduler = build_optimizer(model, args)
        model, optimizer = acc.prepare(model, optimizer)
        assert_fsdp2_ready(model, optimizer, acc)
        model.train()
        _probe_sharded(model, acc)
        checksums = all_gather_objects(trainable_checksum(model))
        if len(set(checksums)) != 1:
            raise RuntimeError("trainable weights differ across ranks before the first update")
        if int(acc.process_index) == 0:
            atomic_kernel_report(args, model, kernel_report)
        resume_manifest = None
        if resume_dir is not None:
            from trim.training.sft_engine.sharded_checkpoint import load_sharded_training_state

            saved = read_json(Path(resume_dir) / "SFT_CHECKPOINT.json")
            if contract is not None:
                require_exact_resume(saved["contract"], contract)
            resume_manifest = load_sharded_training_state(
                Path(resume_dir),
                model=acc.unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                contract=contract if contract is not None else saved["contract"],
                rank=int(acc.process_index),
            )
        else:
            seed_rank(int(args.seed), int(acc.process_index))
        return SimpleNamespace(
            backend=name,
            accelerator=acc,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            contract=contract,
            resume_manifest=resume_manifest,
        )

    from trim.training.sft_engine.model import (
        load_ddp_model,
        make_accelerator,
        probe_action_logits,
    )

    acc = accelerator if accelerator is not None else make_accelerator()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE") or 1)
    local_rank = int(getattr(acc, "local_process_index", 0))
    model = None
    for slot in range(local_world):
        if local_rank == slot:
            if progress is not None:
                progress("load_model", slot=slot)
            model = load_ddp_model(
                str(args.model_name),
                adapter_path,
                acc,
                attn_impl=attn_impl,
                gradient_checkpointing=grad_ckpt == "non_reentrant",
            )
        acc.wait_for_everyone()
    if model is None:
        raise RuntimeError("this rank did not load a model")
    probe_action_logits(model, acc.device)
    checksums = all_gather_objects(trainable_checksum(model))
    if len(set(checksums)) != 1:
        raise RuntimeError("trainable weights differ across ranks before the optimizer is created")
    if int(acc.process_index) == 0:
        atomic_kernel_report(args, model, kernel_report)
    resume_manifest = None
    if resume_dir is not None:
        from trim.training.sft_engine.checkpoint import load_v2_training_state

        saved = read_json(Path(resume_dir) / "SFT_CHECKPOINT.json")
        if contract is not None:
            require_exact_resume(saved["contract"], contract)
        from trim.training.sft_ablation.checkpoint import assert_trainable_signature

        assert_trainable_signature(model, list(saved["trainable"]))
        optimizer, scheduler = build_optimizer(model, args)
        model, optimizer = acc.prepare(model, optimizer)
        resume_manifest = load_v2_training_state(
            Path(resume_dir),
            model=acc.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            contract=contract if contract is not None else saved["contract"],
            rank=int(acc.process_index),
        )
    else:
        optimizer, scheduler = build_optimizer(model, args)
        model, optimizer = acc.prepare(model, optimizer)
        seed_rank(int(args.seed), int(acc.process_index))
    return SimpleNamespace(
        backend=name,
        accelerator=acc,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
        resume_manifest=resume_manifest,
    )


def atomic_kernel_report(args: Any, model: Any, kernel_report: Callable[[Any], dict[str, Any]]) -> None:
    from trim.training.sft_ablation.io_util import atomic_write_json

    try:
        out = Path(args.out)
    except Exception:
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "KERNELS.json", kernel_report(model))
    except Exception as exc:  # diagnostics must not kill the run
        print(f"[sft-backend] kernel report failed: {type(exc).__name__}: {exc}", flush=True)


def _probe_sharded(model: Any, acc: Any) -> None:
    """Startup logits probe after sharding; forward is collective, run together."""
    import torch

    from trim.training.sft_engine.model import probe_action_logits

    with torch.no_grad(), acc.autocast():
        probe_action_logits(model, acc.device)


def update(
    ctx: Any,
    rows: Any,
    max_length: int,
    *,
    grad_clip: float = 1.0,
    trace: Any = None,
    sync_cuda: bool = False,
) -> dict[str, Any]:
    from trim.training.sft_engine.accelerate_engine import update_global_batch

    return update_global_batch(
        ctx.accelerator,
        ctx.model,
        ctx.optimizer,
        ctx.scheduler,
        rows,
        int(max_length),
        grad_clip=float(grad_clip),
        trace=trace,
        sync_cuda=sync_cuda,
    )


def save_training_checkpoint(ctx: Any, directory: Path, **meta: Any) -> Path:
    if ctx.backend == "accelerate_fsdp2":
        from trim.training.sft_engine.sharded_checkpoint import publish_sharded_checkpoint

        return publish_sharded_checkpoint(
            ctx.accelerator,
            Path(directory),
            model=ctx.model,
            optimizer=ctx.optimizer,
            scheduler=ctx.scheduler,
            **meta,
        )
    from trim.training.sft_engine.checkpoint import publish_v2_checkpoint

    return publish_v2_checkpoint(
        ctx.accelerator,
        Path(directory),
        model=ctx.model,
        optimizer=ctx.optimizer,
        scheduler=ctx.scheduler,
        **meta,
    )


def restore_training_checkpoint(ctx: Any, directory: Path, *, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = contract if contract is not None else getattr(ctx, "contract", None)
    if resolved is None:
        raise ValueError("restore_training_checkpoint requires the live run contract")
    if ctx.backend == "accelerate_fsdp2":
        from trim.training.sft_engine.sharded_checkpoint import load_sharded_training_state

        return load_sharded_training_state(
            Path(directory),
            model=ctx.accelerator.unwrap_model(ctx.model),
            optimizer=ctx.optimizer,
            scheduler=ctx.scheduler,
            contract=resolved,
            rank=int(ctx.accelerator.process_index),
        )
    from trim.training.sft_engine.checkpoint import load_v2_training_state

    return load_v2_training_state(
        Path(directory),
        model=ctx.accelerator.unwrap_model(ctx.model),
        optimizer=ctx.optimizer,
        scheduler=ctx.scheduler,
        contract=resolved,
        rank=int(ctx.accelerator.process_index),
    )


def export_policy_adapter(ctx: Any, destination: Path) -> None:
    """Inference/eval artifact: the LoRA adapter only, never the frozen base."""
    acc = ctx.accelerator
    if ctx.backend == "accelerate_fsdp2":
        from trim.training.sft_engine.sharded_checkpoint import export_lora_fsdp2

        export_lora_fsdp2(acc, ctx.model, Path(destination))
        return
    if int(acc.process_index) == 0:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        acc.unwrap_model(ctx.model).save_pretrained(destination)
    acc.wait_for_everyone()
