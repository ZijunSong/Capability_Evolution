"""Episode budgets.

Action / token / extra-LLM-call / wall-clock budgets.  Format errors consume
action budget.  Protocol-level resend is reserved for *network* errors only
and is counted separately from model decision errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.errors import BudgetExceeded


@dataclass
class BudgetSpec:
    max_actions: int = 20
    max_extra_llm_calls: int = 2
    max_actor_output_tokens: int = 4096
    max_tool_calls: int = 40
    max_wall_ms: float = 600_000.0
    # Network-level resend allowance (NOT for model decision errors).
    max_network_retries: int = 2

    def digest(self) -> str:
        return digest_of(
            {
                "max_actions": self.max_actions,
                "max_extra_llm_calls": self.max_extra_llm_calls,
                "max_actor_output_tokens": self.max_actor_output_tokens,
                "max_tool_calls": self.max_tool_calls,
                "max_wall_ms": self.max_wall_ms,
                "max_network_retries": self.max_network_retries,
            }
        )

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "max_actions": self.max_actions,
            "max_extra_llm_calls": self.max_extra_llm_calls,
            "max_actor_output_tokens": self.max_actor_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_wall_ms": self.max_wall_ms,
            "max_network_retries": self.max_network_retries,
        }


@dataclass
class BudgetLedger:
    spec: BudgetSpec
    actions_used: int = 0
    extra_llm_calls_used: int = 0
    actor_output_tokens_used: int = 0
    tool_calls_used: int = 0
    network_retries_used: int = 0
    started_ms: float = field(default_factory=lambda: time.perf_counter() * 1000.0)

    def remaining_actions(self) -> int:
        return max(0, self.spec.max_actions - self.actions_used)

    def charge_action(self, n: int = 1) -> None:
        self.actions_used += n
        if self.actions_used > self.spec.max_actions:
            raise BudgetExceeded(f"action budget exhausted ({self.spec.max_actions})")

    def charge_extra_llm(self, n: int = 1) -> None:
        self.extra_llm_calls_used += n
        if self.extra_llm_calls_used > self.spec.max_extra_llm_calls:
            raise BudgetExceeded(f"extra-LLM-call budget exhausted ({self.spec.max_extra_llm_calls})")

    def charge_actor_output_tokens(self, n: int) -> None:
        self.actor_output_tokens_used += n
        if self.actor_output_tokens_used > self.spec.max_actor_output_tokens:
            raise BudgetExceeded(
                f"actor output token budget exhausted ({self.spec.max_actor_output_tokens})"
            )

    def charge_tool_call(self, n: int = 1) -> None:
        self.tool_calls_used += n
        if self.tool_calls_used > self.spec.max_tool_calls:
            raise BudgetExceeded(f"tool-call budget exhausted ({self.spec.max_tool_calls})")

    def charge_network_retry(self) -> None:
        self.network_retries_used += 1
        if self.network_retries_used > self.spec.max_network_retries:
            raise BudgetExceeded("network retry budget exhausted")

    def check_wall(self) -> None:
        elapsed = time.perf_counter() * 1000.0 - self.started_ms
        if elapsed > self.spec.max_wall_ms:
            raise BudgetExceeded(f"wall clock budget exhausted ({self.spec.max_wall_ms}ms)")

    def snapshot(self) -> dict[str, int]:
        return {
            "actions_used": self.actions_used,
            "extra_llm_calls_used": self.extra_llm_calls_used,
            "actor_output_tokens_used": self.actor_output_tokens_used,
            "tool_calls_used": self.tool_calls_used,
            "network_retries_used": self.network_retries_used,
        }
