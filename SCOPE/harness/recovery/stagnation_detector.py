"""Deterministic stagnation / failure event detection (X1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.capability.rollback_operation import RollbackReasonCode


@dataclass
class FailureEvent:
    reason_code: RollbackReasonCode
    detail: str
    suggested_checkpoint_id: str | None = None


class StagnationDetector:
    def __init__(
        self,
        *,
        no_progress_turns: int = 3,
        query_loop_window: int = 4,
    ) -> None:
        self.no_progress_turns = no_progress_turns
        self.query_loop_window = query_loop_window
        self._evidence_counts: list[int] = []
        self._recent_queries: list[str] = []

    def observe_turn(
        self,
        env: Any,
        *,
        tool_error: str | None = None,
        checkpoint_store: Any | None = None,
    ) -> FailureEvent | None:
        wm = env.wm
        n_curated = len(wm.curated_ids)
        self._evidence_counts.append(n_curated)
        last_q = ""
        if wm.search_history:
            last_q = str(wm.search_history[-1]).strip().lower()
        if last_q:
            self._recent_queries.append(last_q)

        if tool_error:
            ck = self._latest_checkpoint(checkpoint_store)
            return FailureEvent(
                RollbackReasonCode.TOOL_FAILURE,
                tool_error,
                suggested_checkpoint_id=ck,
            )

        if len(self._recent_queries) >= 2 and self._recent_queries[-1] == self._recent_queries[-2]:
            ck = self._latest_checkpoint(checkpoint_store)
            return FailureEvent(
                RollbackReasonCode.QUERY_LOOP,
                "repeated_query",
                suggested_checkpoint_id=ck,
            )

        if len(self._recent_queries) >= self.query_loop_window:
            window = self._recent_queries[-self.query_loop_window:]
            if len(set(window)) <= 2:
                ck = self._latest_checkpoint(checkpoint_store)
                return FailureEvent(
                    RollbackReasonCode.QUERY_LOOP,
                    "query_loop_window",
                    suggested_checkpoint_id=ck,
                )

        if len(self._evidence_counts) >= self.no_progress_turns:
            tail = self._evidence_counts[-self.no_progress_turns:]
            if tail[0] == tail[-1]:
                ck = self._latest_checkpoint(checkpoint_store)
                return FailureEvent(
                    RollbackReasonCode.NO_PROGRESS,
                    f"no_growth_{self.no_progress_turns}_turns",
                    suggested_checkpoint_id=ck,
                )

        remaining = int(getattr(env, "max_turns", 35)) - int(getattr(env, "_current_turn", 0))
        if remaining <= 2 and n_curated < 2:
            ck = self._latest_checkpoint(checkpoint_store)
            return FailureEvent(
                RollbackReasonCode.BUDGET_TRAP,
                "low_budget_branch",
                suggested_checkpoint_id=ck,
            )

        return None

    def _latest_checkpoint(self, checkpoint_store: Any | None) -> str | None:
        if checkpoint_store is None:
            return None
        ids = checkpoint_store.list_ids()
        if not ids:
            return None
        return ids[-1]
