"""Episode advantage, constant-reward filter, and global microbatch planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from trim.training.hf_rl_batch import HF_LENGTH_BUCKET, iter_length_microbatches
from trim.training.hf_rl_opd_client import episode_relative_advantages
from trim.training.dist_runtime import shard_for_rank


def expand_episode_rows(rl_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Broadcast episode advantage onto turn rows; do not GRPO-normalize turns."""
    rows = [dict(r) for r in rl_rows]
    if not rows:
        return []
    if all(row.get("advantage") is not None for row in rows):
        return rows
    adv = episode_relative_advantages(rows)
    out = []
    for row, a in zip(rows, adv):
        item = dict(row)
        item["advantage"] = float(a)
        out.append(item)
    return out


def drop_constant_reward_groups(groups: Sequence[Any]) -> tuple[list[Any], int]:
    """Keep OPD groups even when RL signal is constant; RL-only drops them."""
    kept: list[Any] = []
    n_drop = 0
    for group in groups:
        rewards = [round(float(r), 6) for r in (getattr(group, "terminal_rewards", None) or [])]
        if rewards and len(set(rewards)) <= 1:
            n_drop += 1
            continue
        kept.append(group)
    return kept, n_drop


def training_rows_from_groups(groups: Sequence[Any]) -> list[dict[str, Any]]:
    from trim.integrations.verl.collector import groups_to_rl_rows

    return expand_episode_rows(groups_to_rl_rows(groups))


def shard_training_rows(
    rows: Sequence[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    return shard_for_rank(list(rows), rank=rank, world_size=world_size)


def global_action_token_count(rows: Sequence[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        if row.get("is_dummy"):
            continue
        mask = list(row.get("action_mask") or row.get("response_loss_mask") or [])
        ids = list(row.get("action_ids") or row.get("response_ids") or [])
        if mask:
            total += sum(1 for m in mask if m)
        else:
            total += len(ids)
    return int(total)


def row_input_length(row: Any) -> int:
    from trim.training.tinker_opd_datum import TinkerOPDDatum

    if isinstance(row, TinkerOPDDatum):
        prompt = list(row.prompt_token_ids or [])
        n_prompt = len(prompt)
        targets = list(row.target_tokens or [])
        resp = targets[n_prompt:] if len(targets) >= n_prompt else targets
        teacher = list(row.teacher_prompt_token_ids or [])
        student_len = n_prompt + len(resp)
        teacher_len = len(teacher) + len(resp) if teacher else 0
        return max(1, student_len, teacher_len)
    if isinstance(row, dict):
        prompt = list(
            row.get("prompt_ids")
            or row.get("effective_prompt_ids")
            or row.get("prompt_token_ids")
            or []
        )
        action = list(row.get("action_ids") or row.get("response_ids") or row.get("target_ids") or [])
        if not action and row.get("target_tokens"):
            tokens = list(row.get("target_tokens") or [])
            action = tokens[len(prompt) :] if len(tokens) >= len(prompt) else tokens
        teacher = list(row.get("teacher_prompt_token_ids") or row.get("teacher_prompt_ids") or [])
        student_len = len(prompt) + len(action)
        teacher_len = len(teacher) + len(action) if teacher else 0
        return max(1, student_len, teacher_len)
    if isinstance(row, (tuple, list)) and len(row) >= 2:
        return max(1, len(row[0]) + len(row[1]))
    return 1


def _is_dummy_row(row: Any) -> bool:
    return bool(isinstance(row, dict) and row.get("is_dummy"))


def chunk_padding_stats(
    chunks: Sequence[Sequence[Any]],
    length_fn: Callable[[Any], int] | None = None,
) -> dict[str, float]:
    """Sequence-padding cost inside microbatches. Dummy rows are excluded."""
    fn = length_fn or row_input_length
    real = 0
    padded = 0
    n_rows = 0
    for chunk in chunks:
        lengths = [int(fn(row)) for row in chunk if not _is_dummy_row(row)]
        if not lengths:
            continue
        mx = max(lengths)
        real += sum(lengths)
        padded += mx * len(lengths)
        n_rows += len(lengths)
    return {
        "real_tokens": float(real),
        "padded_tokens": float(padded),
        "padding_fraction": (1.0 - (real / padded)) if padded else 0.0,
        "n_rows": float(n_rows),
    }


def dummy_rl_row() -> dict[str, Any]:
    return {
        "is_dummy": True,
        "row_id": "dummy",
        "query_id": "__dummy__",
        "episode_id": "__dummy__",
        "effective_prompt_ids": [1, 2, 3, 4],
        "prompt_ids": [1, 2, 3, 4],
        "action_ids": [1],
        "token_logprobs": [0.0],
        "action_mask": [0],
        "advantage": 0.0,
    }


def dummy_opd_row() -> dict[str, Any]:
    return {
        "is_dummy": True,
        "prompt_ids": [1, 2, 3, 4],
        "effective_prompt_ids": [1, 2, 3, 4],
        "target_ids": [1],
        "weights": [0.0],
    }


def global_opd_weight_sum(rows: Sequence[Any]) -> float:
    from trim.training.tinker_opd_datum import TinkerOPDDatum

    total = 0.0
    for raw in rows:
        if isinstance(raw, dict) and raw.get("is_dummy"):
            continue
        if isinstance(raw, TinkerOPDDatum):
            n_p = len(list(raw.prompt_token_ids))
            weights = list(raw.weights[n_p:])
        elif isinstance(raw, dict):
            weights = list(raw.get("weights") or [])
        else:
            weights = []
        total += sum(float(w) for w in weights)
    return float(total)


@dataclass
class RankBatchPlan:
    rl_chunks: list[list[dict[str, Any]]] = field(default_factory=list)
    opd_chunks: list[list[Any]] = field(default_factory=list)
    n_rl_rounds: int = 0
    n_opd_rounds: int = 0
    n_dummy_rl: int = 0
    n_dummy_opd: int = 0
    n_global_rl_microbatches: int = 0
    n_global_opd_microbatches: int = 0
    global_rl_tokens: float = 0.0
    global_opd_weight: float = 0.0
    rank: int = 0
    world_size: int = 1
    rl_real_tokens: int = 0
    rl_padded_tokens: int = 0
    rl_padding_fraction: float = 0.0
    opd_real_tokens: int = 0
    opd_padded_tokens: int = 0
    opd_padding_fraction: float = 0.0

    @property
    def n_sync_rounds(self) -> int:
        return int(self.n_rl_rounds) + int(self.n_opd_rounds)

    def steps(self) -> list[tuple[str, list[Any]]]:
        out: list[tuple[str, list[Any]]] = [("rl", chunk) for chunk in self.rl_chunks]
        out.extend(("opd", chunk) for chunk in self.opd_chunks)
        return out


def _assign_chunks_to_ranks(
    chunks: Sequence[Sequence[Any]],
    *,
    world_size: int,
    length_fn: Callable[[Any], int],
    dummy_factory: Callable[[], Any],
) -> tuple[list[list[list[Any]]], int]:
    world = max(1, int(world_size))
    packs = [list(chunk) for chunk in chunks]
    rank_chunks: list[list[list[Any]]] = [[] for _ in range(world)]
    if not packs:
        return rank_chunks, 0
    costs = [sum(length_fn(item) for item in chunk) for chunk in packs]
    rank_cost = [0] * world
    for idx in sorted(range(len(packs)), key=lambda i: (-costs[i], i)):
        dest = min(range(world), key=lambda r: (rank_cost[r], len(rank_chunks[r]), r))
        rank_chunks[dest].append(packs[idx])
        rank_cost[dest] += costs[idx]
    n_rounds = max(len(part) for part in rank_chunks)
    n_dummy = 0
    for part in rank_chunks:
        while len(part) < n_rounds:
            part.append([dummy_factory()])
            n_dummy += 1
    return rank_chunks, n_dummy


def plan_sync_microbatches(
    rows: Sequence[Any],
    *,
    world_size: int,
    micro_batch_size: int,
    length_fn: Callable[[Any], int] | None = None,
    dummy_factory: Callable[[], Any] | None = None,
    bucket: int = HF_LENGTH_BUCKET,
) -> tuple[list[list[list[Any]]], int, int]:
    """Global length-bucket microbatches, then a W-wide sync plan with dummy pads."""
    tagged = []
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            item = dict(row)
            item.setdefault("row_id", i)
            tagged.append(item)
        else:
            tagged.append(row)
    fn = length_fn or row_input_length
    global_mbs = list(
        iter_length_microbatches(tagged, size=max(1, int(micro_batch_size)), length_fn=fn, bucket=bucket)
    )
    assigned, n_dummy = _assign_chunks_to_ranks(
        global_mbs,
        world_size=world_size,
        length_fn=fn,
        dummy_factory=dummy_factory or dummy_rl_row,
    )
    return assigned, n_dummy, len(global_mbs)


def plan_joint_sync_batches(
    rl_rows: Sequence[dict[str, Any]],
    opd_rows: Sequence[Any] | None,
    *,
    rank: int,
    world_size: int,
    micro_batch_size: int,
) -> RankBatchPlan:
    world = max(1, int(world_size))
    rnk = int(rank)
    if rnk < 0 or rnk >= world:
        raise ValueError(f"rank={rnk} is outside world_size={world}")
    rl_assigned, n_dummy_rl, n_global_rl = plan_sync_microbatches(
        list(rl_rows),
        world_size=world,
        micro_batch_size=micro_batch_size,
        dummy_factory=dummy_rl_row,
    )
    opd_assigned, n_dummy_opd, n_global_opd = plan_sync_microbatches(
        list(opd_rows or []),
        world_size=world,
        micro_batch_size=micro_batch_size,
        dummy_factory=dummy_opd_row,
    )
    rl_pad = chunk_padding_stats([chunk for part in rl_assigned for chunk in part], row_input_length)
    opd_pad = chunk_padding_stats([chunk for part in opd_assigned for chunk in part], row_input_length)
    return RankBatchPlan(
        rl_chunks=list(rl_assigned[rnk]),
        opd_chunks=list(opd_assigned[rnk]),
        n_rl_rounds=max((len(p) for p in rl_assigned), default=0),
        n_opd_rounds=max((len(p) for p in opd_assigned), default=0),
        n_dummy_rl=int(n_dummy_rl),
        n_dummy_opd=int(n_dummy_opd),
        n_global_rl_microbatches=int(n_global_rl),
        n_global_opd_microbatches=int(n_global_opd),
        global_rl_tokens=float(global_action_token_count(rl_rows)),
        global_opd_weight=float(global_opd_weight_sum(opd_rows or [])),
        rank=rnk,
        world_size=world,
        rl_real_tokens=int(rl_pad["real_tokens"]),
        rl_padded_tokens=int(rl_pad["padded_tokens"]),
        rl_padding_fraction=float(rl_pad["padding_fraction"]),
        opd_real_tokens=int(opd_pad["real_tokens"]),
        opd_padded_tokens=int(opd_pad["padded_tokens"]),
        opd_padding_fraction=float(opd_pad["padding_fraction"]),
    )
