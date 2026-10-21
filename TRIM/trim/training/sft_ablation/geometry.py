"""4-node, 32-card Scheme A layout used by zero-rl / all-trim.

One coordinator process per node (``nproc_per_node=1``). Each node runs
``rollout_replicas`` independent env workers, one GPU group each. Rank 0 is
the only gradient rank. 4 x 8 x tensor-parallel 1 = 32 cards.
"""

from __future__ import annotations

import os
from typing import Sequence

OFFICIAL_NNODES = 4
OFFICIAL_NPROC_PER_NODE = 1
OFFICIAL_ROLLOUT_REPLICAS = 8
OFFICIAL_TENSOR_PARALLEL_SIZE = 1
OFFICIAL_WORLD_CARDS = 32

EXPERIMENT_MASTER_PORT = {
    "teacher_sft": 29611,
    "projected_teacher_sft": 29612,
    "student_onpolicy_sft": 29613,
}

# v2 train is a normal torchrun world: 8 ranks per node unless the command says otherwise.
V2_NPROC_PER_NODE = 8
V2_MASTER_PORT = {
    "teacher_sft": 29621,
    "projected_teacher_sft": 29622,
    "student_onpolicy_sft": 29623,
}


def world_cards(nnodes: int, replicas: int, tensor_parallel: int) -> int:
    return int(nnodes) * int(replicas) * int(tensor_parallel)


def flag_present(argv: Sequence[str], *names: str) -> bool:
    for token in argv:
        head = str(token).split("=", 1)[0]
        if head in names:
            return True
    return False


def replica_gpu_plan(
    visible_ids: Sequence[str],
    *,
    replicas: int,
    tensor_parallel: int,
) -> list[list[str]]:
    """Disjoint CUDA ids for each replica. Refuses overlapping or short plans."""
    replicas = int(replicas)
    width = int(tensor_parallel)
    if replicas < 1 or width < 1:
        raise ValueError("replicas and tensor_parallel must be positive")
    need = replicas * width
    ids = [str(item) for item in visible_ids]
    if len(ids) < need:
        raise ValueError(
            f"need {need} visible GPUs for replicas={replicas} tp={width}, saw {ids or 'none'}"
        )
    if len(set(ids[:need])) != need:
        raise ValueError(f"replica GPU ids are not unique: {ids[:need]}")
    return [ids[i * width : (i + 1) * width] for i in range(replicas)]


def planned_visible_ids(replicas: int, tensor_parallel: int) -> list[str]:
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    count = world_cards(1, replicas, tensor_parallel)
    return [str(i) for i in range(count)]
