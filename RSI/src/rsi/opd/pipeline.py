"""OPD collection pipeline: Teacher fork -> alignment -> scoring -> rows.

Used by both the real distillability probe and the training collector
(``align_teacher_forks`` in the joint-update flow).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.events import Event
from rsi.contracts.task import PublicTask
from rsi.contracts.trace import DecisionRecord, EpisodeTrace
from rsi.harness.loader import LoadedBank
from rsi.opd.alignment import AlignmentContext, align_decision
from rsi.opd.provenance import collect_fact_refs_by_visibility
from rsi.opd.records import OPDRow, build_opd_row
from rsi.opd.recipes import Recipe
from rsi.opd.scoring import TargetScore, score_aligned_target
from rsi.opd.teacher import TeacherBranch, TeacherForkRunner


@dataclass
class DecisionAlignmentRecord:
    decision: DecisionRecord
    alignment_route: str
    target_kind: str
    opd_row: OPDRow | None
    teacher_branch_ok: bool
    reject_reason: str | None = None


@dataclass
class OPDCollectionStats:
    total_decisions: int = 0
    triggered_decisions: int = 0
    teacher_fork_success: int = 0
    explicit_maps: int = 0
    native_anchors: int = 0
    skips: int = 0
    runtime_only: int = 0
    legal_targets: int = 0
    teacher_scored_targets: int = 0
    effective_weight_tokens: int = 0
    task_clusters: set[str] = field(default_factory=set)
    supported_clusters: set[str] = field(default_factory=set)
    extra_cost_ms: float = 0.0

    def coverage(self) -> dict[str, float]:
        return {
            "legal_targets/all_student_decisions": self.legal_targets / max(1, self.total_decisions),
            "legal_targets/triggered_decisions": self.legal_targets / max(1, self.triggered_decisions),
            "supported_task_clusters/all_task_clusters": (
                len(self.supported_clusters) / max(1, len(self.task_clusters))
            ),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "triggered_decisions": self.triggered_decisions,
            "teacher_fork_success": self.teacher_fork_success,
            "explicit_maps": self.explicit_maps,
            "native_anchors": self.native_anchors,
            "skips": self.skips,
            "runtime_only": self.runtime_only,
            "legal_targets": self.legal_targets,
            "teacher_scored_targets": self.teacher_scored_targets,
            "effective_weight_tokens": self.effective_weight_tokens,
            "task_clusters": len(self.task_clusters),
            "supported_task_clusters": len(self.supported_clusters),
            "extra_cost_ms": self.extra_cost_ms,
            "coverage": self.coverage(),
        }


def collect_opd_rows(
    *,
    policy: Any,
    teacher_policy: Any,
    env: Any,
    bank: LoadedBank,
    task: PublicTask,
    trace: EpisodeTrace,
    recipes: Sequence[Recipe] = (),
    uncertain_realizability: str = "skip",
    max_targets_per_decision: int = 1,
    max_shadow_events: int = 8,
    teacher_runner: TeacherForkRunner | None = None,
    sample_decisions: Sequence[int] | None = None,
) -> tuple[list[DecisionAlignmentRecord], OPDCollectionStats]:
    """Align Teacher branches for the decisions of one Student trace."""
    runner = teacher_runner or TeacherForkRunner(
        env_factory=type(env),
        policy=policy,
        bank=bank,
        max_shadow_events=max_shadow_events,
    )
    stats = OPDCollectionStats()
    out: list[DecisionAlignmentRecord] = []
    catalog = env.tool_catalog()
    ctx = AlignmentContext(
        catalog=catalog,
        state_view={},
        uncertain_realizability=uncertain_realizability,
        max_targets_per_decision=max_targets_per_decision,
    )
    decisions = list(trace.decisions)
    wanted = set(sample_decisions) if sample_decisions is not None else None

    for i, decision in enumerate(decisions):
        stats.total_decisions += 1
        if wanted is not None and i not in wanted:
            continue
        stats.triggered_decisions += 1
        stats.task_clusters.add(task.group_id)
        # Teacher fork at this decision's snapshot.
        try:
            snapshot = env.snapshot()
            branch = runner.run_branch(
                base_env=env,
                snapshot_ref=snapshot,
                task=task,
                student_events=trace.events,
                decision_index=i,
            )
        except Exception:  # noqa: BLE001 - fork failures are recorded, not fatal
            stats.skips += 1
            out.append(
                DecisionAlignmentRecord(
                    decision=decision,
                    alignment_route="skip",
                    target_kind="none",
                    opd_row=None,
                    teacher_branch_ok=False,
                    reject_reason="teacher_fork_failed",
                )
            )
            continue
        stats.teacher_fork_success += 1
        stats.extra_cost_ms += branch.cost.wall_ms

        student_events = trace.events  # full public prefix at decision i
        student_refs, _ = collect_fact_refs_by_visibility(student_events)
        facts = {r: r for r in student_refs}
        alignment = align_decision(
            teacher_events=branch.events,
            student_events=student_events,
            student_pre_state_digest=decision.public_view_digest,
            ctx=ctx,
            recipes=recipes,
            recipe_step_index=0,
            visible_facts=facts,
        )
        if alignment.route == "explicit_map":
            stats.explicit_maps += 1
        elif alignment.route == "native_anchor":
            stats.native_anchors += 1
        elif alignment.route == "runtime_only":
            stats.runtime_only += 1
        else:
            stats.skips += 1

        if alignment.target_action is None:
            out.append(
                DecisionAlignmentRecord(
                    decision=decision,
                    alignment_route=alignment.route,
                    target_kind=alignment.target_kind,
                    opd_row=None,
                    teacher_branch_ok=True,
                    reject_reason=alignment.reject_reason,
                )
            )
            continue

        score = score_aligned_target(
            policy=policy,
            teacher_policy=teacher_policy,
            task=task,
            catalog=catalog,
            teacher_prefix_events=branch.events,
            student_prompt_text="",  # exact ids from the record are used below
            alignment=alignment,
            student_prompt_token_ids=decision.student_prompt_token_ids,
        )
        if score is None:
            stats.skips += 1
            out.append(
                DecisionAlignmentRecord(
                    decision=decision,
                    alignment_route=alignment.route,
                    target_kind=alignment.target_kind,
                    opd_row=None,
                    teacher_branch_ok=True,
                    reject_reason="no_legal_teacher_scoring_view",
                )
            )
            continue

        stats.legal_targets += 1
        stats.teacher_scored_targets += 1
        stats.effective_weight_tokens += len(score.target_token_ids)
        stats.supported_clusters.add(task.group_id)
        row = build_opd_row(decision, score, alignment)
        out.append(
            DecisionAlignmentRecord(
                decision=decision,
                alignment_route=alignment.route,
                target_kind=score.target_kind,
                opd_row=row,
                teacher_branch_ok=True,
            )
        )
    return out, stats
