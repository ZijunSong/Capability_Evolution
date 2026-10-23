"""Job identity and score-cache invalidation (task book §10.4).

``JobKey`` covers policy, bank, H0, environment/image, split/panel, seed,
decoding, budget, scorer, projector and config digests.  Candidate file
changes change the hash — cached scores never leak across them.  Any policy
change invalidates all score caches (round 2 re-measures the H0 baseline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.contracts.common import digest_of
from rsi.contracts.policy import RunIdentity


@dataclass(frozen=True)
class JobKey:
    identity: RunIdentity

    @property
    def key(self) -> str:
        return self.identity.job_key()


class ScoreCache:
    """Content-addressed cache.  Entries are keyed by the full JobKey digest;
    a changed policy/bank/config produces a different key automatically."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, job_key: JobKey) -> dict[str, Any] | None:
        p = self._path(job_key.key)
        if p.is_file():
            return json.loads(p.read_text("utf-8"))
        return None

    def put(self, job_key: JobKey, result: dict[str, Any]) -> None:
        self._path(job_key.key).write_text(
            json.dumps(result, sort_keys=True, ensure_ascii=False), "utf-8"
        )

    def invalidate_policy(self, policy_digest: str) -> int:
        """Drop every entry whose payload references the given policy digest."""
        n = 0
        for p in self.root.glob("*.json"):
            try:
                payload = json.loads(p.read_text("utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("policy_digest") == policy_digest:
                p.unlink()
                n += 1
        return n


def write_job_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
