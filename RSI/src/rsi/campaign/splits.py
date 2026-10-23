"""Split roles, confirm-panel consumption and leakage guards.

Confirm panels (``confirm_h[k]`` / ``confirm_w[k]``) are consumed exactly once
per round.  Looking at confirm failures and then editing against the same
panel, or retrying confirm_w until it passes, is recorded as consumption
violation and refuses.  ``final_test`` is unreachable before campaign freeze.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.contracts.errors import ProtocolViolation


@dataclass
class ConsumptionLedger:
    path: Path

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            return json.loads(self.path.read_text("utf-8"))
        return {"consumed": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")

    def is_consumed(self, key: str) -> bool:
        return key in self._load()["consumed"]

    def consume(self, key: str, *, context: str, allow_reuse: bool = False) -> None:
        """Record one consumption of a confirm panel.

        Idempotent re-entry with the SAME context (resume after interruption of
        the same stage) is not a second attempt.  A different context — e.g.
        looking at confirm_w failures and retrying promotion decisions — is a
        violation unless ``allow_reuse`` (pilot_reused_dev) declares it.
        """
        data = self._load()
        consumed = data["consumed"]
        if key in consumed:
            prev = consumed[key]
            if prev.get("context") == context:
                prev["reentries"] = int(prev.get("reentries", 0)) + 1
            elif allow_reuse:
                prev.setdefault("reuse_count", 0)
                prev["reuse_count"] += 1
                prev.setdefault("reuse_contexts", []).append(context)
            else:
                raise ProtocolViolation(
                    f"confirm panel {key} already consumed by {prev.get('context')!r}; "
                    f"refusing second consumption from {context!r}"
                )
        else:
            consumed[key] = {"context": context, "attempts": 1}
        self._save(data)

    def consumed_keys(self) -> list[str]:
        return sorted(self._load()["consumed"])


def assert_role_allowed(role: str, *, campaign_state: str, frozen: bool) -> None:
    if role == "final_test" and not (frozen or campaign_state in ("FINAL_EVAL", "DONE")):
        raise ProtocolViolation(
            "final_test is sealed until campaign freeze; evaluation before freeze "
            "uses weight_dev/confirm panels only"
        )
