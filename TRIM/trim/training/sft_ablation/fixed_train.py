"""Fixed-dataset CE updates for Teacher SFT and Projected Teacher SFT."""

from __future__ import annotations

import random
from typing import Sequence

from trim.training.sft_ablation.contracts import SFTBatchRow


def take_batch(
    rows: Sequence[SFTBatchRow],
    *,
    cursor: int,
    batch_size: int,
    seed: int,
) -> tuple[list[SFTBatchRow], int, dict[str, int]]:
    if not rows:
        raise ValueError("no SFT rows")
    ordered = sorted(rows, key=lambda row: row.row_id)
    count = len(ordered)
    picked: list[SFTBatchRow] = []
    uses: dict[str, int] = {}
    epoch_orders: dict[int, list[int]] = {}
    for offset in range(int(batch_size)):
        index = int(cursor) + offset
        epoch = index // count
        if epoch not in epoch_orders:
            rng = random.Random(f"{int(seed)}:{epoch}")
            perm = list(range(count))
            rng.shuffle(perm)
            epoch_orders[epoch] = perm
        row = ordered[epoch_orders[epoch][index % count]]
        picked.append(row)
        uses[row.row_id] = uses.get(row.row_id, 0) + 1
    return picked, int(cursor) + int(batch_size), uses
