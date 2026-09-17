"""Epoch-based query sampling before rollout.

Rollout only the query groups selected for the current optimizer step.
In-batch query IDs are unique. Cross-epoch fill records deferred items
instead of duplicating a query inside the same batch.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


def pool_fingerprint(pool: Sequence[Any]) -> str:
    ids = [_query_id(r, i) for i, r in enumerate(pool)]
    blob = json.dumps({"query_ids": ids, "n": len(ids)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sample_params_fingerprint(*, base_seed: int, groups_per_step: int) -> str:
    blob = json.dumps(
        {"base_seed": int(base_seed), "groups_per_step": int(groups_per_step)},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    attempt_id: int = 0
    pool_fingerprint: str = ""
    sample_params_hash: str = ""
    deferred_indices: list[int] = field(default_factory=list)

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
        current_fp = pool_fingerprint(self.pool)
        current_params = sample_params_fingerprint(
            base_seed=self.base_seed, groups_per_step=self.groups_per_step
        )
        if state is not None:
            self.state = state
            if self.state.pool_size != len(self.pool):
                raise ValueError(
                    f"sampler pool_size={self.state.pool_size} != len(pool)={len(self.pool)}"
                )
            if self.state.pool_fingerprint and self.state.pool_fingerprint != current_fp:
                raise ValueError(
                    "sampler pool fingerprint mismatch: checkpoint and live query pool differ"
                )
            if self.state.sample_params_hash and self.state.sample_params_hash != current_params:
                raise ValueError(
                    "sampler sample-params fingerprint mismatch "
                    f"(seed/groups_per_step) checkpoint={self.state.sample_params_hash} live={current_params}"
                )
            if int(self.state.base_seed) != self.base_seed:
                raise ValueError(
                    f"sampler base_seed={self.state.base_seed} != live={self.base_seed}"
                )
            if int(self.state.groups_per_step) != self.groups_per_step:
                raise ValueError(
                    f"sampler groups_per_step={self.state.groups_per_step} != live={self.groups_per_step}"
                )
            if not self.state.pool_fingerprint:
                self.state.pool_fingerprint = current_fp
            if not self.state.sample_params_hash:
                self.state.sample_params_hash = current_params
        else:
            self.state = QuerySamplerState(
                base_seed=self.base_seed,
                groups_per_step=self.groups_per_step,
                pool_size=len(self.pool),
                pool_fingerprint=current_fp,
                sample_params_hash=current_params,
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
        deferred: list[int] = []
        if self.state.cursor < len(self.state.order):
            take = min(want, len(self.state.order) - self.state.cursor)
            picked.extend(self.state.order[self.state.cursor : self.state.cursor + take])
            self.state.cursor += take
        if len(picked) < want:
            self.state.epoch += 1
            self._reshuffle_epoch()
            picked_set = set(picked)
            leftover: list[int] = []
            for idx in list(self.state.order):
                if len(picked) >= want:
                    leftover.append(idx)
                    continue
                if idx in picked_set:
                    deferred.append(idx)
                    continue
                picked.append(idx)
                picked_set.add(idx)
            self.state.order = deferred + leftover
            self.state.cursor = 0
        if len(set(picked)) != len(picked):
            raise RuntimeError(f"in-batch duplicate query indices: {picked}")
        self.state.deferred_indices = list(deferred)
        chosen = [self.pool[i] for i in picked]
        ids = [_query_id(r, i) for i, r in zip(picked, chosen)]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"in-batch duplicate query_ids: {ids}")
        return chosen, _meta(
            sampled=True,
            n_groups=len(chosen),
            n_pool=len(self.pool),
            query_ids=ids,
            state=self.state,
            picked_indices=picked,
            deferred_indices=deferred,
        )

    def note_rollout_start(self) -> None:
        self.state.global_rollout_batch += 1
        self.state.attempt_id += 1

    def note_update_complete(self) -> None:
        self.state.global_optimizer_step += 1


def _query_id(row: Any, idx: int) -> str:
    if isinstance(row, dict):
        return str(row.get("query_id", idx))
    return str(getattr(row, "query_id", idx))


def _meta(
    *,
    sampled: bool,
    n_groups: int,
    n_pool: int,
    query_ids: list[str],
    state: QuerySamplerState,
    picked_indices: list[int] | None = None,
    deferred_indices: list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sampled": sampled,
        "n_groups": n_groups,
        "n_pool": n_pool,
        "query_ids": query_ids,
        "epoch": state.epoch,
        "cursor": state.cursor,
        "global_optimizer_step": state.global_optimizer_step,
        "global_rollout_batch": state.global_rollout_batch,
        "attempt_id": state.attempt_id,
        "seed": state.base_seed + state.global_optimizer_step,
        "pool_fingerprint": state.pool_fingerprint,
        "sample_params_hash": state.sample_params_hash,
        "unique_in_batch": True,
    }
    if picked_indices is not None:
        payload["picked_indices"] = picked_indices
    if deferred_indices is not None:
        payload["deferred_indices"] = deferred_indices
    return payload
