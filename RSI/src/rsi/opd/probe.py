"""Real distillability probe (task book §8.6).

Statistics are collected on real tasks at real Student decision points: total
decisions, triggers, Teacher fork successes, explicit maps, anchors, legal
targets, teacher-scored targets, effective weighted tokens, covered clusters,
OPD gradient contribution and extra cost.  Coverage ratios are reported as
``legal_targets/all_student_decisions``, ``legal_targets/triggered_decisions``
and ``supported_task_clusters/all_task_clusters``.

Explicit-projection scarcity does NOT mean a B-class component is useless —
native-anchor supervision counts as effective coverage.  No fixed 10% coverage
threshold is enforced (task/decision densities are incomparable); formal
thresholds are frozen after a pilot.  The first-version *engineering* admission
floor: non-zero real OPD samples, non-zero effective supervised tokens, and a
non-zero finite OPD gradient — proof the chain works, not proof of scale
benefit.  Structural triggers never satisfy admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsi.contracts.common import JSONValue
from rsi.opd.losses import gated_projected_gap_loss
from rsi.opd.pipeline import OPDCollectionStats, collect_opd_rows
from rsi.opd.records import OPDRow


@dataclass
class ProbeReport:
    stats: OPDCollectionStats
    opd_grad_norm: float | None
    n_rows: int
    admitted: bool
    admission_reasons: tuple[str, ...]
    synthetic: bool = False

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "stats": self.stats.to_json(),
            "opd_grad_norm": self.opd_grad_norm,
            "n_rows": self.n_rows,
            "admitted": self.admitted,
            "admission_reasons": list(self.admission_reasons),
            "synthetic": self.synthetic,
        }

    @property
    def has_verified_support(self) -> bool:
        return self.admitted


def opd_gradient_contribution(rows: Sequence[OPDRow], *, gate_beta: float = 5.0) -> float | None:
    """Gradient norm of the OPD objective w.r.t. student target logprobs.

    Uses the real loss on the real aligned rows (pure-function gradient — the
    model-side gradient is validated separately in training acceptance tests).
    Returns None when there is no supervision at all.
    """
    import torch

    if not rows:
        return None
    total = 0.0
    saw_token = False
    for row in rows:
        n = len(row.target_token_ids)
        if n == 0 or not row.student_logprobs or not row.teacher_logprobs:
            continue
        saw_token = True
        student_lp = torch.tensor(row.student_logprobs, requires_grad=True)
        teacher_lp = torch.tensor(row.teacher_logprobs)
        mask = torch.tensor(row.target_mask, dtype=torch.float32)
        weight = torch.tensor(row.token_weight or [1.0] * n, dtype=torch.float32)
        loss = gated_projected_gap_loss(student_lp, teacher_lp, mask, weight, gate_beta=gate_beta)
        loss.backward()
        grad = student_lp.grad
        if grad is None:
            continue
        total += float(grad.abs().sum().item())
    if not saw_token:
        return None
    return total


def admit_probe(stats: OPDCollectionStats, opd_grad_norm: float | None) -> tuple[bool, tuple[str, ...]]:
    """Engineering admission floor only (see module docstring)."""
    reasons: list[str] = []
    if stats.legal_targets <= 0 or stats.teacher_scored_targets <= 0:
        reasons.append("no_real_opd_samples")
    if stats.effective_weight_tokens <= 0:
        reasons.append("no_effective_supervised_tokens")
    if opd_grad_norm is None or not (opd_grad_norm > 0.0) or not (opd_grad_norm < float("inf")):
        reasons.append("opd_grad_not_nonzero_finite")
    return (not reasons, tuple(reasons))


def run_probe_on_traces(
    *,
    policy: Any,
    teacher_policy: Any,
    env_factory: Any,
    bank: Any,
    tasks: Sequence[Any],
    trace_runner: Any,
    gate_beta: float = 5.0,
    uncertain_realizability: str = "skip",
    max_targets_per_decision: int = 1,
    max_shadow_events: int = 8,
    sampling: Any | None = None,
    budgets: Any | None = None,
    synthetic: bool = False,
) -> ProbeReport:
    """Run real Student rollouts, Teacher forks and alignment on a probe panel."""
    from rsi.opd.teacher import TeacherForkRunner

    runner = TeacherForkRunner(
        env_factory=env_factory,
        policy=policy,
        bank=bank,
        max_shadow_events=max_shadow_events,
        sampling=sampling,
        budgets=budgets,
    )
    agg = OPDCollectionStats()
    all_rows: list[OPDRow] = []
    for task in tasks:
        env = env_factory()
        trace = trace_runner(env, task)
        records, stats = collect_opd_rows(
            policy=policy,
            teacher_policy=teacher_policy,
            env=env,
            bank=bank,
            task=task,
            trace=trace,
            uncertain_realizability=uncertain_realizability,
            max_targets_per_decision=max_targets_per_decision,
            max_shadow_events=max_shadow_events,
            teacher_runner=runner,
        )
        agg.total_decisions += stats.total_decisions
        agg.triggered_decisions += stats.triggered_decisions
        agg.teacher_fork_success += stats.teacher_fork_success
        agg.explicit_maps += stats.explicit_maps
        agg.native_anchors += stats.native_anchors
        agg.skips += stats.skips
        agg.runtime_only += stats.runtime_only
        agg.legal_targets += stats.legal_targets
        agg.teacher_scored_targets += stats.teacher_scored_targets
        agg.effective_weight_tokens += stats.effective_weight_tokens
        agg.task_clusters |= stats.task_clusters
        agg.supported_clusters |= stats.supported_clusters
        agg.extra_cost_ms += stats.extra_cost_ms
        all_rows.extend(r.opd_row for r in records if r.opd_row is not None)
        env.close()

    grad_norm = opd_gradient_contribution(all_rows, gate_beta=gate_beta)
    admitted, reasons = admit_probe(agg, grad_norm)
    return ProbeReport(
        stats=agg,
        opd_grad_norm=grad_norm,
        n_rows=len(all_rows),
        admitted=admitted,
        admission_reasons=reasons,
        synthetic=synthetic,
    )
