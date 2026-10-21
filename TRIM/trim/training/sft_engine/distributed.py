"""Collectives used by the v2 engine. The process group belongs to Accelerate."""

from __future__ import annotations

from typing import Any


def broadcast_ints(values: list[int] | None, *, device: Any, src: int = 0) -> list[int]:
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        if values is None:
            raise RuntimeError("broadcast source did not provide indices")
        return [int(item) for item in values]
    rank = dist.get_rank()
    if rank == src:
        data = torch.tensor([int(item) for item in (values or [])], dtype=torch.long, device=device)
        length = torch.tensor([int(data.numel())], dtype=torch.long, device=device)
    else:
        length = torch.zeros(1, dtype=torch.long, device=device)
        data = None
    dist.broadcast(length, src=src)
    if rank != src:
        data = torch.empty(int(length.item()), dtype=torch.long, device=device)
    dist.broadcast(data, src=src)
    return [int(item) for item in data.tolist()]


def all_gather_objects(value: Any) -> list[Any]:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return [value]
    box: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(box, value)
    return box
