"""Equal-slot plans for one global CE update.

Dummy slots keep every rank inside the same number of forward/backward
calls. They are plan attributes, not rows, and they contribute nothing
to the supervised mass.
"""

from __future__ import annotations

from typing import Sequence


def plan_slots(global_rows: Sequence, world_size: int) -> list[list[int | None]]:
    if world_size < 1:
        raise ValueError("Invalid world_size")
    if not global_rows:
        return [[] for _ in range(world_size)]
    order = sorted(
        range(len(global_rows)),
        key=lambda i: (
            -(len(global_rows[i].prompt_ids) + len(global_rows[i].response_ids)),
            i,
        ),
    )
    nslots = (len(order) + world_size - 1) // world_size
    padded = list(order) + [None] * (nslots * world_size - len(order))
    ranks: list[list[int | None]] = [[] for _ in range(world_size)]
    for step in range(nslots):
        block = padded[step * world_size : (step + 1) * world_size]
        if step % 2:
            block = list(reversed(block))
        for rank, index in enumerate(block):
            ranks[rank].append(index)
    return ranks


def assert_slot_plan(ranks: Sequence[Sequence[int | None]], n_rows: int, world_size: int) -> None:
    """Real indices are a permutation of the batch; dummies only pad the rectangle."""
    if len(ranks) != world_size:
        raise AssertionError(f"expected {world_size} ranks, saw {len(ranks)}")
    if n_rows == 0:
        if any(rank for rank in ranks):
            raise AssertionError("empty batch produced slots")
        return
    widths = {len(rank) for rank in ranks}
    if len(widths) != 1:
        raise AssertionError(f"uneven slot counts: {sorted(widths)}")
    seen: list[int] = []
    for rank in ranks:
        for index in rank:
            if index is None:
                continue
            if index < 0 or index >= n_rows:
                raise AssertionError(f"index out of range: {index}")
            seen.append(index)
    if sorted(seen) != list(range(n_rows)):
        raise AssertionError("plan dropped or duplicated a real row")
