"""Metered RuntimeAPI handed to harness components."""

from __future__ import annotations

import time
from typing import Any, Mapping

from rsi.contracts.action import Action, ToolCall
from rsi.contracts.common import JSONValue
from rsi.contracts.errors import BudgetExceeded
from rsi.contracts.events import CostRecord
from rsi.contracts.observation import Observation
from rsi.runtime.budgets import BudgetLedger


class MeteredAPI:
    """Concrete RuntimeAPI.

    All extra component spend (tool calls, auxiliary model calls) is charged to
    the episode budget and recorded; components cannot forge logs or costs.
    Component tool results are private to the component (they are not appended
    to the Student-visible history by the runtime).
    """

    def __init__(
        self,
        *,
        tool_runner,  # Callable[[Action], Observation]
        model_caller,  # Callable[[str, int], str]
        ledger: BudgetLedger,
        visibility: str = "teacher_private",
    ):
        self._tool_runner = tool_runner
        self._model_caller = model_caller
        self._ledger = ledger
        self._visibility = visibility
        self._state: dict[str, Any] = {}
        self._logs: list[tuple[str, dict[str, JSONValue]]] = []
        self._cost = CostRecord()

    # -- RuntimeAPI ------------------------------------------------------
    def call_tool(self, call: ToolCall) -> Observation:
        self._ledger.charge_tool_call(1)
        t0 = time.perf_counter()
        obs = self._tool_runner(call)
        self._cost = CostRecord(
            actor_input_tokens=self._cost.actor_input_tokens,
            actor_output_tokens=self._cost.actor_output_tokens,
            aux_input_tokens=self._cost.aux_input_tokens,
            aux_output_tokens=self._cost.aux_output_tokens,
            tool_calls=self._cost.tool_calls + 1,
            extra_llm_calls=self._cost.extra_llm_calls,
            wall_ms=self._cost.wall_ms + (time.perf_counter() - t0) * 1000.0,
        )
        return obs

    def call_model(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        self._ledger.charge_extra_llm(1)
        t0 = time.perf_counter()
        out = self._model_caller(prompt, max_new_tokens)
        self._cost = CostRecord(
            actor_input_tokens=self._cost.actor_input_tokens,
            actor_output_tokens=self._cost.actor_output_tokens,
            aux_input_tokens=self._cost.aux_input_tokens + max(1, len(prompt) // 4),
            aux_output_tokens=self._cost.aux_output_tokens + max(1, len(out) // 4),
            tool_calls=self._cost.tool_calls,
            extra_llm_calls=self._cost.extra_llm_calls + 1,
            wall_ms=self._cost.wall_ms + (time.perf_counter() - t0) * 1000.0,
        )
        return out

    def get_state(self) -> Mapping[str, Any]:
        return dict(self._state)

    def set_state(self, patch: Mapping[str, JSONValue]) -> None:
        for key, value in dict(patch).items():
            if isinstance(value, dict):
                bucket = self._state.setdefault(key, {})
                if isinstance(bucket, dict):
                    bucket.update(value)
                else:
                    self._state[key] = value
            else:
                self._state[key] = value

    def budget(self) -> Mapping[str, int]:
        return self._ledger.snapshot()

    def log(self, message: str, **fields: JSONValue) -> None:
        self._logs.append((message, dict(fields)))

    # -- introspection ----------------------------------------------------
    @property
    def cost(self) -> CostRecord:
        return self._cost

    @property
    def logs(self) -> list[tuple[str, dict[str, JSONValue]]]:
        return list(self._logs)
