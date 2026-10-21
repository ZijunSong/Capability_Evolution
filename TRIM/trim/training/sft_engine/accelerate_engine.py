"""One global weighted CE update on a DDP world. Manual mass, one optimizer step."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Callable, Sequence

from trim.training.sft_ablation.contracts import supervised_weight
from trim.training.sft_engine.action_loss import action_ce_numerator
from trim.training.sft_engine.planner import plan_slots


def all_sum(value: Any) -> Any:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def update_global_batch(
    acc: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    global_rows: Sequence[Any],
    max_length: int,
    *,
    grad_clip: float = 1.0,
    trace: Callable[..., None] | None = None,
    sync_cuda: bool = False,
) -> dict[str, Any]:
    """Backward ``W * N_rk / Z`` on every slot, then one AdamW step on every rank.

    ``Z`` is the supervised mass of this whole optimizer update. Dummy slots
    keep the step count identical and add nothing to ``Z``. Accelerate's own
    accumulation stays at 1 so it does not scale the loss a second time.
    """
    import torch
    import torch.distributed as dist

    if int(getattr(acc, "gradient_accumulation_steps", 1)) != 1:
        raise ValueError("Manual global normalization requires accumulation_steps=1")
    world, rank = int(acc.num_processes), int(acc.process_index)
    for row in global_rows:
        row.validate(max_length)
    if not global_rows:
        return {
            "updated": False,
            "loss_ce": None,
            "grad_norm": None,
            "n_rows": 0,
            "weighted_tokens": 0.0,
            "local_slots": 0,
            "gradient_ranks": world,
            "dummy_rows": 0,
            "real_rows": 0,
        }
    slots = plan_slots(global_rows, world)[rank]
    local_mass = sum(supervised_weight(global_rows[index]) for index in slots if index is not None)
    local_real = sum(1 for index in slots if index is not None)
    local_dummy = sum(1 for index in slots if index is None)
    local_input = sum(
        len(global_rows[index].prompt_ids) + len(global_rows[index].response_ids)
        for index in slots
        if index is not None
    )
    local_supervised = sum(
        sum(1 for bit in global_rows[index].response_mask if bit) for index in slots if index is not None
    )
    z = all_sum(torch.tensor(local_mass, dtype=torch.float64, device=acc.device))
    if not torch.isfinite(z) or float(z.item()) <= 0:
        raise ValueError("Invalid global supervised mass")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    local_numerator = torch.zeros((), dtype=torch.float64, device=acc.device)
    reference = min(global_rows, key=lambda row: len(row.prompt_ids) + len(row.response_ids))
    forward_s = 0.0
    backward_s = 0.0
    for slot_id, row_index in enumerate(slots):
        dummy = row_index is None
        row = reference if dummy else global_rows[row_index]
        if trace is not None:
            # NOTE: do NOT pass global_step here. fixed_runtime's trace lambda
            # injects global_step=step+1 into every event; passing it here too
            # raises "RankProgress.event() got multiple values for keyword
            # argument 'global_step'" on the first microbatch (g5yjb failed
            # this way at 12:32:15). The lambda owns global_step; events only
            # carry their own fields.
            trace(
                "microbatch_start",
                slot=slot_id,
                total_slots=len(slots),
                dummy=dummy,
            )
        context = acc.no_sync(model) if slot_id + 1 < len(slots) else nullcontext()
        _sync(acc, sync_cuda)
        started = time.perf_counter()
        with context:
            with acc.autocast():
                numerator = action_ce_numerator(model, row, device=acc.device, dummy=dummy)
            _sync(acc, sync_cuda)
            forward_s += time.perf_counter() - started
            backward_started = time.perf_counter()
            acc.backward(numerator * (world / z.to(dtype=numerator.dtype)))
            _sync(acc, sync_cuda)
            backward_s += time.perf_counter() - backward_started
        local_numerator = local_numerator + numerator.detach().double()
        del numerator
        if trace is not None:
            trace(
                "microbatch_end",
                slot=slot_id,
                total_slots=len(slots),
                dummy=dummy,
                forward_s=forward_s,
                backward_s=backward_s,
            )
    clip_started = time.perf_counter()
    norm = acc.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], float(grad_clip))
    _sync(acc, sync_cuda)
    clip_s = time.perf_counter() - clip_started
    sync_started = time.perf_counter()
    total_numerator = all_sum(local_numerator)
    good = torch.tensor(
        int(torch.isfinite(norm).item() and torch.isfinite(total_numerator).item()),
        device=acc.device,
        dtype=torch.int32,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(good, op=dist.ReduceOp.MIN)
    _sync(acc, sync_cuda)
    sync_s = time.perf_counter() - sync_started
    if not int(good.item()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite CE/grad: no rank may optimizer.step")
    opt_started = time.perf_counter()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    _sync(acc, sync_cuda)
    optimizer_s = time.perf_counter() - opt_started
    input_tokens = all_sum(torch.tensor(local_input, dtype=torch.float64, device=acc.device))
    supervised_tokens = all_sum(torch.tensor(local_supervised, dtype=torch.float64, device=acc.device))
    real_rows = all_sum(torch.tensor(local_real, dtype=torch.float64, device=acc.device))
    dummy_rows = all_sum(torch.tensor(local_dummy, dtype=torch.float64, device=acc.device))
    return {
        "updated": True,
        "loss_ce": float((total_numerator / z).item()),
        "grad_norm": float(norm.item()),
        "n_rows": len(global_rows),
        "weighted_tokens": float(z.item()),
        "local_slots": len(slots),
        "gradient_ranks": world,
        "forward_s": forward_s,
        "loss_s": 0.0,
        "backward_s": backward_s,
        "clip_s": clip_s,
        "sync_s": sync_s,
        "optimizer_s": optimizer_s,
        "input_tokens": float(input_tokens.item()),
        "supervised_tokens": float(supervised_tokens.item()),
        "real_rows": int(real_rows.item()),
        "dummy_rows": int(dummy_rows.item()),
    }


def _sync(acc: Any, enabled: bool) -> None:
    if not enabled:
        return
    import torch

    if acc.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(acc.device)
