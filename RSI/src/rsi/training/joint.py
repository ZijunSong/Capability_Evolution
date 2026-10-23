"""Joint RL/OPD update orchestration (task book §9.2/§9.3).

Method switches:

===================  ===  ===  ============  ============  ====
method               RL   OPD  explicit_map  native_anchor gate
===================  ===  ===  ============  ============  ====
harness_opd_rl       on   on   on            on            on
rl_only              on   off  --            --            --
opd_only             off  on   on            on            on
no_project           on   on   off           on            on
no_gate              on   on   on            on            const-1
aligned_teacher_sft  off  CE   (sft data)    (sft data)    n/a
===================  ===  ===  ============  ============  ====

``no_project`` is not "no alignment at all"; ``no_gate`` keeps legality,
provenance and isolation checks with gate constant 1 (``beta=0`` yields 0.5 and
must NOT be reported as no_gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsi.contracts.errors import NotImplementedCapability, ProtocolViolation
from rsi.opd.records import OPDRow, RLRow
from rsi.training.backend import GlobalNormalizers, TrainingBackend

METHODS = (
    "harness_opd_rl",
    "rl_only",
    "opd_only",
    "no_project",
    "no_gate",
    "aligned_teacher_sft",
)


@dataclass(frozen=True)
class MethodSpec:
    method: str
    use_rl: bool
    use_opd: bool
    explicit_map: bool
    native_anchor: bool
    gate_mode: str  # sigmoid | none | n/a

    @staticmethod
    def resolve(method: str) -> "MethodSpec":
        table = {
            "harness_opd_rl": MethodSpec(method, True, True, True, True, "sigmoid"),
            "rl_only": MethodSpec(method, True, False, False, False, "n/a"),
            "opd_only": MethodSpec(method, False, True, True, True, "sigmoid"),
            "no_project": MethodSpec(method, True, True, False, True, "sigmoid"),
            "no_gate": MethodSpec(method, True, True, True, True, "none"),
            "aligned_teacher_sft": MethodSpec(method, False, True, True, True, "n/a"),
        }
        if method not in table:
            raise ProtocolViolation(f"unknown training method {method!r}; known: {METHODS}")
        return table[method]

    @property
    def routes_enabled(self) -> tuple[str, ...]:
        routes = []
        if self.explicit_map:
            routes.append("explicit_map")
        if self.native_anchor:
            routes.append("native_anchor")
        return tuple(routes)


@dataclass
class JointStepReport:
    n_rl_rows: int
    n_opd_rows: int
    n_optimizer_steps: int
    rl_metrics: Mapping[str, float]
    opd_metrics: Mapping[str, float]
    step_metrics: Mapping[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "n_rl_rows": self.n_rl_rows,
            "n_opd_rows": self.n_opd_rows,
            "n_optimizer_steps": self.n_optimizer_steps,
            "rl_metrics": dict(self.rl_metrics),
            "opd_metrics": dict(self.opd_metrics),
            "step_metrics": dict(self.step_metrics),
        }


def joint_update(
    backend: TrainingBackend,
    *,
    rl_rows: Sequence[RLRow],
    opd_rows: Sequence[OPDRow],
    spec: MethodSpec,
    lambda_opd: float = 0.01,
    world_size: int = 1,
) -> JointStepReport:
    """One joint optimizer step over one batch.

    Both gradients accumulate before a single ``optimizer_step``
    (``test_joint_step_count``).  Empty OPD is a zero contribution plus a
    recorded reason, never a dummy target.
    """
    if spec.method == "aligned_teacher_sft":
        raise NotImplementedCapability(
            "aligned_teacher_sft is a separate SFT pipeline (CE on aligned targets), "
            "not a rename of HarnessOPD; invoke the SFT entry explicitly"
        )
    if spec.use_opd and not getattr(backend, "consumes_opd", False):
        raise NotImplementedCapability(
            f"backend {getattr(backend, 'name', '?')!r} does not consume OPD; "
            "refusing to silently degrade to RL-only"
        )
    normalizers = GlobalNormalizers.from_rows(rl_rows, opd_rows, world_size=world_size)
    backend.zero_grad()
    rl_metrics: dict[str, float] = {}
    opd_metrics: dict[str, float] = {}
    if spec.use_rl and rl_rows:
        rl_metrics = backend.backward_rl(rl_rows, normalizers)
    if spec.use_opd:
        if opd_rows:
            opd_metrics = backend.backward_opd(opd_rows, normalizers, lambda_opd=lambda_opd)
        else:
            opd_metrics = {"opd_skipped": 1.0, "reason_code": 0.0}  # 0 = no supervised rows
    step_metrics = backend.optimizer_step()
    return JointStepReport(
        n_rl_rows=len(rl_rows),
        n_opd_rows=len(opd_rows) if spec.use_opd else 0,
        n_optimizer_steps=1,
        rl_metrics=rl_metrics,
        opd_metrics=opd_metrics,
        step_metrics=step_metrics,
    )
