"""Cost accounting rollup (task book §11.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from rsi.contracts.events import CostRecord


@dataclass
class CostLedger:
    task_success: int = 0
    protocol_valid: int = 0
    completed: int = 0
    n_tasks: int = 0
    base_tool_calls: int = 0
    extra_tool_calls: int = 0
    actor_input_tokens: int = 0
    actor_output_tokens: int = 0
    aux_input_tokens: int = 0
    aux_output_tokens: int = 0
    wall_ms: float = 0.0
    wall_p95_ms: float = 0.0
    teacher_shadow_cost: CostRecord = field(default_factory=CostRecord)
    proposer_cost: CostRecord = field(default_factory=CostRecord)
    train_gpu_hours: float = 0.0
    infra_failures: int = 0
    model_failures: int = 0

    def add_episode(
        self,
        *,
        success: bool,
        protocol_valid: bool,
        completed: bool,
        cost: CostRecord,
        wall_ms: float,
        infra_failure: bool = False,
        model_failure: bool = False,
    ) -> None:
        self.n_tasks += 1
        self.task_success += int(success)
        self.protocol_valid += int(protocol_valid)
        self.completed += int(completed)
        self.base_tool_calls += cost.tool_calls
        self.actor_input_tokens += cost.actor_input_tokens
        self.actor_output_tokens += cost.actor_output_tokens
        self.aux_input_tokens += cost.aux_input_tokens
        self.aux_output_tokens += cost.aux_output_tokens
        self.wall_ms += wall_ms
        self.infra_failures += int(infra_failure)
        self.model_failures += int(model_failure)

    def to_json(self) -> dict[str, object]:
        return {
            "n_tasks": self.n_tasks,
            "task_success": self.task_success,
            "protocol_valid": self.protocol_valid,
            "completed": self.completed,
            "base_tool_calls": self.base_tool_calls,
            "extra_tool_calls": self.extra_tool_calls,
            "actor_input_tokens": self.actor_input_tokens,
            "actor_output_tokens": self.actor_output_tokens,
            "aux_input_tokens": self.aux_input_tokens,
            "aux_output_tokens": self.aux_output_tokens,
            "wall_ms_total": self.wall_ms,
            "wall_p95_ms": self.wall_p95_ms,
            "teacher_shadow_cost": self.teacher_shadow_cost.to_json(),
            "proposer_cost": self.proposer_cost.to_json(),
            "train_gpu_hours": self.train_gpu_hours,
            "infra_failures": self.infra_failures,
            "model_failures": self.model_failures,
        }
