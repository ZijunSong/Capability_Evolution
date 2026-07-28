"""JSONL dataset IO for OPDTransitionV2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from training.opd_v2.transitions import OPDTransitionV2


def write_transitions_jsonl(
    path: str | Path,
    transitions: Iterable[OPDTransitionV2],
    *,
    append: bool = False,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with path.open(mode, encoding="utf-8") as f:
        for tr in transitions:
            f.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_transitions_jsonl(path: str | Path) -> list[OPDTransitionV2]:
    path = Path(path)
    out: list[OPDTransitionV2] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(OPDTransitionV2.from_dict(json.loads(line)))
    return out


def iter_transitions_jsonl(path: str | Path) -> Iterator[OPDTransitionV2]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield OPDTransitionV2.from_dict(json.loads(line))


class TransitionBuffer:
    """In-memory buffer for online training."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self._items: list[OPDTransitionV2] = []

    def add(self, transition: OPDTransitionV2) -> None:
        self._items.append(transition)
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]

    def extend(self, transitions: Iterable[OPDTransitionV2]) -> None:
        for t in transitions:
            self.add(t)

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[OPDTransitionV2]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

    def by_mode(self, mode: str) -> list[OPDTransitionV2]:
        return [t for t in self._items if t.mode.value == mode and t.validity_mask]
