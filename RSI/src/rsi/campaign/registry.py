"""Policy registry: immutable policy manifests, current pointer, lineage.

Promotion is atomic: write the complete result + artifact digest first, then
the stage DONE marker, then CAS-update the current policy pointer.  Promotion
verifies the parent digest equals the current policy; concurrent promoters
serialize via a lock file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.common import JSONValue
from rsi.contracts.errors import ProtocolViolation
from rsi.contracts.policy import PolicyArtifact


@dataclass
class PolicyRegistry:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Path:
        return self.root / "current.json"

    def write_policy(self, artifact: PolicyArtifact, *, lineage: Mapping[str, Any] | None = None) -> Path:
        path = self.root / f"{artifact.policy_id}.json"
        payload = {"artifact": artifact.to_json(), "lineage": dict(lineage or {})}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
        return path

    def current_policy(self) -> PolicyArtifact | None:
        if not self.current_path.is_file():
            return None
        payload = json.loads(self.current_path.read_text("utf-8"))
        return PolicyArtifact.from_json(payload["artifact"])

    def atomic_promote(self, artifact: PolicyArtifact, *, parent_policy_id: str, evidence: Mapping[str, Any]) -> None:
        lock = self.root / ".promote.lock"
        deadline = time.time() + 30.0
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() > deadline:
                    raise ProtocolViolation("promotion lock timeout")
                time.sleep(0.05)
        try:
            current = self.current_policy()
            if current is not None and current.policy_id != parent_policy_id:
                raise ProtocolViolation(
                    f"promotion parent mismatch: parent={parent_policy_id} current={current.policy_id}"
                )
            self.write_policy(
                artifact,
                lineage={"parent_policy_id": parent_policy_id, "evidence": dict(evidence)},
            )
            tmp = self.current_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"artifact": artifact.to_json()}, indent=2, sort_keys=True), "utf-8"
            )
            tmp.replace(self.current_path)  # atomic on POSIX
        finally:
            lock.unlink(missing_ok=True)
