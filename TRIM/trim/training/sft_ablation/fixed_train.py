"""Fixed-dataset CE updates for Teacher SFT and Projected Teacher SFT."""

from __future__ import annotations

import random
from typing import Sequence

from trim.training.sft_ablation.contracts import SFTBatchRow

_ORDER_CACHE: dict[tuple, list[SFTBatchRow]] = {}
_PERM_CACHE: dict[tuple, list[int]] = {}


def epoch_permutation(n_rows: int, epoch: int, seed: int) -> list[int]:
    """Permutation of ``0 .. n_rows-1`` for one epoch. Cached for the process."""
    key = (int(seed), int(epoch), int(n_rows))
    found = _PERM_CACHE.get(key)
    if found is None:
        rng = random.Random(f"{int(seed)}:{int(epoch)}")
        found = list(range(int(n_rows)))
        rng.shuffle(found)
        _PERM_CACHE[key] = found
    return found


def select_batch_indices(
    n_rows: int,
    *,
    cursor: int,
    batch_size: int,
    seed: int,
) -> tuple[list[int], int, dict[int, int]]:
    """Same cursor rule as ``take_batch``, returning positions in row_id order."""
    if int(n_rows) < 1:
        raise ValueError("no SFT rows")
    picked: list[int] = []
    uses: dict[int, int] = {}
    for offset in range(int(batch_size)):
        index = int(cursor) + offset
        epoch = index // int(n_rows)
        choice = epoch_permutation(n_rows, epoch, seed)[index % int(n_rows)]
        picked.append(choice)
        uses[choice] = uses.get(choice, 0) + 1
    return picked, int(cursor) + int(batch_size), uses


def take_batch(
    rows: Sequence[SFTBatchRow],
    *,
    cursor: int,
    batch_size: int,
    seed: int,
) -> tuple[list[SFTBatchRow], int, dict[str, int]]:
    if not rows:
        raise ValueError("no SFT rows")
    ordered = _ordered_rows(rows)
    logical, new_cursor, _uses = select_batch_indices(
        len(ordered),
        cursor=cursor,
        batch_size=batch_size,
        seed=seed,
    )
    picked: list[SFTBatchRow] = []
    uses: dict[str, int] = {}
    for choice in logical:
        row = ordered[choice]
        picked.append(row)
        uses[row.row_id] = uses.get(row.row_id, 0) + 1
    return picked, new_cursor, uses


def _ordered_rows(rows: Sequence[SFTBatchRow]) -> list[SFTBatchRow]:
    key = (id(rows), len(rows), rows[0].row_id, rows[-1].row_id)
    found = _ORDER_CACHE.get(key)
    if found is None:
        found = sorted(rows, key=lambda row: row.row_id)
        _ORDER_CACHE[key] = found
    return found
