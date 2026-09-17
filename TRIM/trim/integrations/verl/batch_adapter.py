"""Episode advantage, constant-reward filter, and rank sharding for FSDP2."""

from __future__ import annotations

from typing import Any, Sequence

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
        mask = list(row.get("action_mask") or row.get("response_loss_mask") or [])
        ids = list(row.get("action_ids") or row.get("response_ids") or [])
        if mask:
            total += sum(1 for m in mask if m)
        else:
            total += len(ids)
    return int(total)
