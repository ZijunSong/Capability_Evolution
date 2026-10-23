"""Training backend protocol (task book §9.3/§9.4).

One joint update per batch by default:

    snapshot = policy.freeze_for_batch()
    rollouts = collector.collect(snapshot, harness=H0, tasks=train_batch)
    rl_rows  = build_rl_rows(rollouts)
    opd_rows = align_teacher_forks(snapshot, bank=Bk, decisions=rollouts.decisions)
    backend.zero_grad()
    backend.backward_rl(rl_rows, global_normalizers)
    backend.backward_opd(opd_rows, global_normalizers)
    backend.optimizer_step()
    new_policy = backend.export_policy()
    actor.reload_and_verify(new_policy)

RL and OPD gradients accumulate into the SAME optimizer step
(``test_joint_step_count``).  Sub-step batching must be pre-registered; RL and
OPD accumulate per sub-step and step together — never 200 steps becoming 400
independent updates.

Backends: ``hf_reference`` (small model, numerical parity, single process) and
``torch_fsdp2`` (production).  A backend that does not consume OPD must fail
loudly, never silently degrade to RL-only (``test_unimplemented_fails``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from rsi.opd.records import OPDRow, RLRow


@dataclass(frozen=True)
class GlobalNormalizers:
    """One global effective token/weight statistic per batch.

    When a gradient reducer world-size-averages, local numerators are scaled by
    the documented convention (see ``opd.losses.scale_projected_gap_loss``);
    a global denominator is never all-reduced twice.
    """

    global_action_tokens: int
    global_opd_weight: float
    world_size: int = 1

    @staticmethod
    def from_rows(rl_rows: Sequence[RLRow], opd_rows: Sequence[OPDRow], world_size: int = 1) -> "GlobalNormalizers":
        action_tokens = sum(sum(r.action_mask) for r in rl_rows)
        opd_weight = 0.0
        for row in opd_rows:
            if row.token_weight:
                opd_weight += sum(row.token_weight[i] * m for i, m in enumerate(row.target_mask))
            else:
                opd_weight += sum(row.target_mask)
        return GlobalNormalizers(
            global_action_tokens=max(1, int(action_tokens)),
            global_opd_weight=float(opd_weight),
            world_size=world_size,
        )


@runtime_checkable
class TrainingBackend(Protocol):
    name: str
    consumes_opd: bool

    def zero_grad(self) -> None: ...

    def backward_rl(self, rows: Sequence[RLRow], normalizers: GlobalNormalizers) -> dict[str, float]: ...

    def backward_opd(
        self, rows: Sequence[OPDRow], normalizers: GlobalNormalizers, *, lambda_opd: float
    ) -> dict[str, float]: ...

    def optimizer_step(self) -> dict[str, float]: ...

    def export_policy(self) -> Any: ...
