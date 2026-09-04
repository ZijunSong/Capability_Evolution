"""Apply / restore Harness-1 component masks via process env (no upstream edits)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping

from trim.adapters.components import (
    all_component_ids,
    flag_for,
    full_mask,
    mask_to_env,
    minus_mask,
)


class ComponentMaskContext:
    """Track applied mask and previously overwritten env values."""

    def __init__(self, mask: Mapping[str, bool]):
        self.mask = dict(mask)
        self._prev: dict[str, str | None] = {}

    def apply(self) -> None:
        env = mask_to_env(self.mask)
        for key, value in env.items():
            self._prev[key] = os.environ.get(key)
            os.environ[key] = value

    def restore(self) -> None:
        for key, prev in self._prev.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
        self._prev.clear()


@contextmanager
def apply_component_mask(mask: Mapping[str, bool]) -> Iterator[dict[str, bool]]:
    ctx = ComponentMaskContext(mask)
    ctx.apply()
    try:
        yield dict(mask)
    finally:
        ctx.restore()


@contextmanager
def full_harness() -> Iterator[dict[str, bool]]:
    with apply_component_mask(full_mask()) as mask:
        yield mask


@contextmanager
def minus_component(component_id: str) -> Iterator[dict[str, bool]]:
    with apply_component_mask(minus_mask(component_id)) as mask:
        yield mask


def current_mask_from_env() -> dict[str, bool]:
    from trim.adapters.components import env_to_mask

    return env_to_mask(os.environ)


def only_toggle(component_id: str, *, enabled: bool, base: Mapping[str, bool] | None = None) -> dict[str, bool]:
    """Return a mask equal to base except for one component."""
    from trim.adapters.harness_profiles import infer_harness_from_ids

    harness = infer_harness_from_ids([component_id])
    if component_id not in all_component_ids(harness):
        raise KeyError(component_id)
    mask = dict(base or full_mask(harness))
    mask[component_id] = enabled
    return mask


def upstream_flag_value(component_id: str) -> str | None:
    return os.environ.get(flag_for(component_id))
