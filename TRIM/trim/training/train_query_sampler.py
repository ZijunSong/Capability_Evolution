"""Epoch-based query sampling before rollout.

Rollout only the query groups selected for the current optimizer step.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class QuerySamplerState:
    base_seed: int
    groups_per_step: int
    pool_size: int
    epoch: int = 0
    cursor: int = 0
    order: list[int] = field(default_factory=list)
    global_optimizer_step: int = 0
    global_rollout_batch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuerySamplerState:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


class QuerySampler:
    """Deterministic epoch shuffle over the full query pool."""

    def __init__(
        self,
        pool: Sequence[Any],
        *,
        base_seed: int,
        groups_per_step: int,
        state: QuerySamplerState | None = None,
    ) -> None:
        self.pool = list(pool)
        self.base_seed = int(base_seed)
        self.groups_per_step = int(groups_per_step)
        if state is not None:
            self.state = state
            if self.state.pool_size != len(self.pool):
                raise ValueError(
                    f"sampler pool_size={self.state.pool_size} != len(pool)={len(self.pool)}"
                )
        else:
            self.state = QuerySamplerState(
                base_seed=self.base_seed,
                groups_per_step=self.groups_per_step,
                pool_size=len(self.pool),
            )
            self._reshuffle_epoch()

    def _reshuffle_epoch(self) -> None:
        rng = random.Random(self.base_seed + self.state.epoch * 9973)
        self.state.order = list(range(len(self.pool)))
        rng.shuffle(self.state.order)
        self.state.cursor = 0

    def sample_for_rollout(self) -> tuple[list[Any], dict[str, Any]]:
        want = self.groups_per_step
        if want <= 0 or want >= len(self.pool):
            ids = [_query_id(r, i) for i, r in enumerate(self.pool)]
            return list(self.pool), _meta(
                sampled=False,
                n_groups=len(self.pool),
                n_pool=len(self.pool),
                query_ids=ids,
                state=self.state,
            )
        picked: list[int] = []
        while len(picked) < want:
            if self.state.cursor >= len(self.state.order):
                self.state.epoch += 1
                self._reshuffle_epoch()
            remain = want - len(picked)
            take = min(remain, len(self.state.order) - self.state.cursor)
            picked.extend(self.state.order[self.state.cursor : self.state.cursor + take])
            self.state.cursor += take
        chosen = [self.pool[i] for i in picked]
        ids = [_query_id(r, i) for i, r in zip(picked, chosen)]
        return chosen, _meta(
            sampled=True,
            n_groups=len(chosen),
            n_pool=len(self.pool),
            query_ids=ids,
            state=self.state,
            picked_indices=picked,
        )

    def note_rollout_start(self) -> None:
        self.state.global_rollout_batch += 1

    def note_update_complete(self) -> None:
        self.state.global_optimizer_step += 1


def _query_id(row: Any, idx: int) -> str:
    if isinstance(row, dict):
        return str(row.get("query_id", idx))
    return str(getattr(row, "query_id", idx))


def _meta(*, sampled: bool, n_groups: int, n_pool: int, query_ids: list[str], state: QuerySamplerState, picked_indices: list[int] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sampled": sampled,
        "n_groups": n_groups,
        "n_pool": n_pool,
        "query_ids": query_ids,
        "epoch": state.epoch,
        "cursor": state.cursor,
        "global_optimizer_step": state.global_optimizer_step,
        "global_rollout_batch": state.global_rollout_batch,
        "seed": state.base_seed + state.global_optimizer_step,
    }
    if picked_indices is not None:
        payload["picked_indices"] = picked_indices
    return payload
