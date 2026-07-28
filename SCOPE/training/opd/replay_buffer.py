"""OPD replay buffer."""

from __future__ import annotations

from collections import deque
from typing import Iterator

from training.opd._policy_backend import OPDTransition


class OPDReplayBuffer:
    def __init__(self, max_size: int = 10_000) -> None:
        self._buffer: deque[OPDTransition] = deque(maxlen=max_size)

    def add(self, transition: OPDTransition) -> None:
        self._buffer.append(transition)

    def extend(self, transitions: list[OPDTransition]) -> None:
        for t in transitions:
            self.add(t)

    def sample(self, batch_size: int) -> list[OPDTransition]:
        items = list(self._buffer)
        if not items:
            return []
        if len(items) <= batch_size:
            return items
        return items[:batch_size]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[OPDTransition]:
        return iter(self._buffer)

    def successful_only(self) -> list[OPDTransition]:
        return [t for t in self._buffer if t.success]
