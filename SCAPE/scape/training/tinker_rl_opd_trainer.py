"""Joint CISPO + SR-OPD trainer.

Does not rewrite CISPO. Two Tinker forward_backward calls accumulate
gradients; a single optim_step applies them.

    L_hybrid = L_CISPO + λ L_OPD
    scape+rl uses SEED gated sampled-token OPD, not projected CE / full-vocab KL.
    scape+seed keeps action projection and applies the same SEED-scale gap on a*.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from scape.training.opd_dataset import (
    ProjectionAudit,
    ProjectedTrainingStep,
    finalize_audit,
    project_and_materialize,
)
from scape.training.opd_events import HarnessEvent
from scape.training.opd_projection import StudentActionSpaceProjector
from scape.training.rl_opd_metrics import empty_hybrid_metrics, mean_reward
from scape.training.rl_opd_policy_version import assert_policy_versions_match
from scape.training.tinker_opd_datum import (
    EncodeFn,
    TinkerOPDDatum,
    build_projected_seed_datums,
    build_sampled_opd_datums,
    build_tinker_opd_datums,
    default_encode,
)
from scape.training.rl_opd_types import (
    UPDATE_OPD_ONLY_ZERO_RL,
    UPDATE_RL_ONLY,
    UPDATE_RL_OPD_JOINT,
    UPDATE_SKIPPED,
    HybridRolloutGroup,
    HybridStepMetrics,
    HybridTrainingBatch,
    SCAPE_RL_OPD_GATE_BETA,
    StudentDecisionPoint,
    uses_sampled_opd,
    uses_seed_gap,
    uses_projected_seed,
)


TeacherEventFn = Callable[[StudentDecisionPoint], list[HarnessEvent]]


def _n_rl_tokens(datums: Sequence[Any]) -> int:
    total = 0
    for row in datums:
        if hasattr(row, "n_tokens"):
            total += int(row.n_tokens)
        elif isinstance(row, dict):
            total += int(row.get("n_tokens") or len(row.get("target_tokens") or []))
        else:
            tokens = getattr(row, "target_tokens", None)
            if tokens is not None:
                total += len(tokens)
    return total


def _n_opd_tokens(datums: Sequence[TinkerOPDDatum]) -> int:
    return int(sum(d.n_supervised_tokens for d in datums))


def classify_update_type(*, n_rl: int, n_opd: int) -> str:
    if n_rl and n_opd:
        return UPDATE_RL_OPD_JOINT
    if n_rl:
        return UPDATE_RL_ONLY
    if n_opd:
        return UPDATE_OPD_ONLY_ZERO_RL
    return UPDATE_SKIPPED


def split_even(items: Sequence[Any], n_parts: int) -> list[list[Any]]:
    """Split into n_parts contiguous chunks. Extra items go to earlier parts."""
    n = max(1, int(n_parts))
    seq = list(items)
    if not seq:
        return [[] for _ in range(n)]
    base, extra = divmod(len(seq), n)
    out: list[list[Any]] = []
    i = 0
    for part in range(n):
        take = base + (1 if part < extra else 0)
        out.append(seq[i : i + take])
        i += take
    return out


def split_hybrid_substeps(
    rl_datums: Sequence[Any],
    opd_datums: Sequence[Any],
    *,
    num_substeps: int = 4,
) -> list[tuple[list[Any], list[Any]]]:
    """One optimizer step per pair. NUM_SUBSTEPS is not doubled by OPD."""
    rl_parts = split_even(rl_datums, num_substeps)
    opd_parts = split_even(opd_datums, num_substeps)
    return list(zip(rl_parts, opd_parts))


def rewards_are_constant(rewards: Sequence[float]) -> bool:
    vals = [float(x) for x in rewards]
    return len(vals) >= 2 and max(vals) - min(vals) < 1e-12


def filter_constant_reward_rl_datums(
    groups: Sequence[HybridRolloutGroup],
    rl_datums_by_query: dict[str, list[Any]],
) -> list[Any]:
    """Drop RL datums whose group reward has no signal. OPD is not dropped."""
    kept: list[Any] = []
    for group in groups:
        if rewards_are_constant(group.terminal_rewards):
            continue
        kept.extend(rl_datums_by_query.get(group.query_id, []))
    return kept


def sample_decision_points(
    points: Sequence[StudentDecisionPoint],
    *,
    per_trajectory: int,
    seed: int,
    include_valid_failures: bool = True,
    include_format_errors: bool = False,
) -> list[StudentDecisionPoint]:
    """Deterministic subset. Failures stay eligible; format errors do not.

    ``per_trajectory < 0`` keeps every eligible action on the trajectory
    (scape+rl). ``per_trajectory == 0`` samples nothing.
    """
    eligible: list[StudentDecisionPoint] = []
    for point in points:
        if not point.structurally_valid:
            if include_format_errors:
                eligible.append(point)
            continue
        if (point.reward or 0.0) <= 0.0 and not include_valid_failures:
            continue
        eligible.append(point)
    if not eligible or per_trajectory == 0:
        return []

    by_ep: dict[str, list[StudentDecisionPoint]] = {}
    for point in eligible:
        by_ep.setdefault(point.episode_id, []).append(point)

    picked: list[StudentDecisionPoint] = []
    take_all = int(per_trajectory) < 0
    for episode_id, rows in sorted(by_ep.items()):
        rng = random.Random(int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest(), 16) % (2**32))

        def score(p: StudentDecisionPoint) -> tuple[int, int]:
            toolish = 1 if any(n not in {"end_search", "user_text"} for n in p.action_tool_names) else 0
            terminalish = 1 if "end_search" in p.action_tool_names else 0
            return (toolish, terminalish)

        ranked = sorted(rows, key=lambda p: (-score(p)[0], -score(p)[1], p.turn_id))
        if take_all or len(ranked) <= per_trajectory:
            chosen = ranked
        else:
            head = ranked[: max(1, per_trajectory // 2)]
            rest = [p for p in ranked if p not in head]
            rng.shuffle(rest)
            chosen = (head + rest)[:per_trajectory]
        picked.extend(sorted(chosen, key=lambda p: (p.episode_id, p.turn_id)))
    return picked


def project_on_policy_decisions(
    points: Sequence[StudentDecisionPoint],
    *,
    teacher_event_fn: TeacherEventFn,
    component_id: str,
    projector: StudentActionSpaceProjector | None = None,
) -> tuple[list[ProjectedTrainingStep], ProjectionAudit, dict[str, Any]]:
    """Teacher is a side branch: events only. RL snapshots are not mutated."""
    proj = projector or StudentActionSpaceProjector()
    audit = ProjectionAudit()
    steps: list[ProjectedTrainingStep] = []
    overlap_hits = 0
    overlap_total = 0
    for point in points:
        events = list(teacher_event_fn(point) or [])
        projection, mat = project_and_materialize(
            student_snapshot=point.pre_action_snapshot,
            teacher_events=events,
            student_mask=point.pre_action_snapshot.harness_mask,
            component_id=component_id,
            projector=proj,
            audit=audit,
        )
        del projection
        for step in mat:
            step.metadata["source_policy_version"] = point.policy_version
            step.metadata["decision_point_id"] = point.decision_point_id
            if point.student_action_text and step.target_text:
                overlap_total += 1
                if point.student_action_text.strip() == step.target_text.strip() or (
                    point.action_tool_names and point.action_tool_names[0] == step.target_action.get("name")
                ):
                    overlap_hits += 1
        steps.extend(mat)
    finalize_audit(audit)
    extras = {
        "rl_opd_exact_target_overlap_rate": (overlap_hits / overlap_total) if overlap_total else 0.0,
        "n_sampled_decision_points": len(points),
    }
    return steps, audit, extras


def prepare_hybrid_batch(
    *,
    groups: Sequence[HybridRolloutGroup],
    rl_datums_by_query: dict[str, list[Any]],
    policy_version: str,
    lambda_opd: float,
    component_id: str,
    teacher_event_fn: TeacherEventFn | None,
    encode_fn: EncodeFn | None = None,
    opd_states_per_trajectory: int = 3,
    seed: int = 0,
    include_valid_failures: bool = True,
    include_format_errors: bool = False,
    remove_constant_reward_groups: bool = True,
    projector: StudentActionSpaceProjector | None = None,
    opd_loss: str = "sr_opd_ce",
    opd_gate_beta: float = SCAPE_RL_OPD_GATE_BETA,
) -> HybridTrainingBatch:
    """Extract OPD states before constant-reward RL filtering.

    ``lambda_opd <= 0`` skips Teacher / projector / OPD datums entirely.
    scape+rl (sampled-gap) scores CISPO sampled actions and does not project.
    scape+seed projects teacher events to student-legal a*, then applies the
    SEED gated token-mean on those projected tokens.
    """
    all_points: list[StudentDecisionPoint] = []
    for group in groups:
        all_points.extend(group.decision_points)

    if remove_constant_reward_groups:
        rl_datums = filter_constant_reward_rl_datums(groups, rl_datums_by_query)
    else:
        rl_datums = [row for group in groups for row in rl_datums_by_query.get(group.query_id, [])]

    projection_stats: dict[str, Any] = {"skipped_teacher": True}
    opd_datums: list[TinkerOPDDatum] = []
    skipped_teacher = True

    if float(lambda_opd) > 0.0:
        sampled = sample_decision_points(
            all_points,
            per_trajectory=opd_states_per_trajectory,
            seed=seed,
            include_valid_failures=include_valid_failures,
            include_format_errors=include_format_errors,
        )
        if uses_sampled_opd(opd_loss):
            opd_datums = build_sampled_opd_datums(
                sampled,
                lambda_opd=lambda_opd,
                encode_fn=encode_fn or default_encode,
                policy_version=policy_version,
                component_id=component_id,
                gate_beta=opd_gate_beta,
                opd_loss=opd_loss,
            )
            skipped_teacher = False
            projection_stats = {
                "skipped_teacher": False,
                "projector_used": False,
                "sampled_action_opd": True,
                "projection_coverage": 0.0,
                "reject_rate": 0.0,
                "n_direct": 0,
                "n_macro": 0,
                "n_reject": 0,
                "n_projected_training_steps": 0,
                "n_sampled_decision_points": len(sampled),
                "n_sampled_opd_datums": len(opd_datums),
                "rl_opd_exact_target_overlap_rate": 1.0,
            }
        else:
            if teacher_event_fn is None:
                raise ValueError("teacher_event_fn is required when lambda_opd > 0")
            steps, audit, extras = project_on_policy_decisions(
                sampled,
                teacher_event_fn=teacher_event_fn,
                component_id=component_id,
                projector=projector,
            )
            if uses_projected_seed(opd_loss):
                opd_datums = build_projected_seed_datums(
                    steps,
                    lambda_opd=lambda_opd,
                    encode_fn=encode_fn or default_encode,
                    policy_version=policy_version,
                    gate_beta=opd_gate_beta,
                    opd_loss=opd_loss,
                )
            else:
                opd_datums = build_tinker_opd_datums(
                    steps,
                    lambda_opd=lambda_opd,
                    encode_fn=encode_fn or default_encode,
                    policy_version=policy_version,
                    opd_loss=opd_loss,
                )
            skipped_teacher = False
            projection_stats = {
                "skipped_teacher": False,
                "projector_used": True,
                "sampled_action_opd": False,
                "projection_coverage": audit.projection_coverage,
                "reject_rate": audit.n_reject / max(1, audit.n_teacher_segments),
                "n_direct": audit.n_direct,
                "n_macro": audit.n_macro,
                "n_reject": audit.n_reject,
                "n_projected_training_steps": audit.n_projected_training_steps,
                **extras,
            }

    rewards = [r for g in groups for r in g.terminal_rewards]
    return HybridTrainingBatch(
        policy_version=policy_version,
        rl_datums=list(rl_datums),
        opd_datums=list(opd_datums),
        n_rl_tokens=_n_rl_tokens(rl_datums),
        n_opd_tokens=_n_opd_tokens(opd_datums),
        projection_stats=projection_stats,
        reward_stats=mean_reward(rewards),
        skipped_teacher=skipped_teacher,
        update_type=classify_update_type(n_rl=len(rl_datums), n_opd=len(opd_datums)),
        opd_loss=str(opd_loss),
    )


def _extract_loss(result: Any) -> float | None:
    if result is None:
        return None
    if isinstance(result, dict):
        for key in ("loss", "nll", "mean_loss"):
            if key in result and result[key] is not None:
                return float(result[key])
    return None


async def hybrid_train_substep(
    *,
    training_client: Any,
    rl_datums: Sequence[Any],
    opd_datums: Sequence[Any],
    rl_loss_fn: str,
    rl_loss_fn_config: dict[str, Any] | None,
    lambda_opd: float,
    adam_params: Any,
    policy_version: str,
    rollout_policy: str | None = None,
    train_policy: str | None = None,
    harness_teacher_policy: str | None = None,
    projection_coverage: float = 0.0,
    reject_rate: float = 0.0,
    overlap_rate: float = 0.0,
    opd_loss: str = "sr_opd_ce",
) -> HybridStepMetrics:
    """RL FB, then OPD FB, then exactly one optim_step.

    For CE OPD, ``lambda_opd`` is baked into token weights. For sampled-gap
    OPD, ``lambda_opd`` lives in datum metadata and is applied as
    ``λ × token-mean`` inside the OPD FB. Callers must not scale again.
    """
    assert_policy_versions_match(
        rollout_policy=rollout_policy or policy_version,
        train_policy=train_policy or policy_version,
        harness_teacher_policy=harness_teacher_policy or policy_version,
        sync=True,
    )
    requested_lambda = float(lambda_opd)

    rl_list = list(rl_datums)
    opd_list = list(opd_datums)
    if not rl_list and not opd_list:
        return empty_hybrid_metrics(policy_version=policy_version, lambda_opd=requested_lambda)

    rl_result = None
    opd_result = None
    n_rl_fb = 0
    n_opd_fb = 0

    if rl_list:
        maybe = training_client.forward_backward_async(
            rl_list,
            loss_fn=rl_loss_fn,
            loss_fn_config=rl_loss_fn_config or {},
        )
        rl_result = await maybe if isinstance(maybe, Awaitable) else maybe
        n_rl_fb = 1

    if opd_list:
        opd_fn = "sampled_gap" if uses_seed_gap(opd_loss) else "cross_entropy"
        maybe = training_client.forward_backward_async(
            opd_list,
            loss_fn=opd_fn,
        )
        opd_result = await maybe if isinstance(maybe, Awaitable) else maybe
        n_opd_fb = 1

    maybe_opt = training_client.optim_step_async(adam_params)
    if isinstance(maybe_opt, Awaitable):
        await maybe_opt

    n_rl_tok = _n_rl_tokens(rl_list)
    n_opd_tok = _n_opd_tokens(opd_list)
    return HybridStepMetrics(
        update_type=classify_update_type(n_rl=len(rl_list), n_opd=len(opd_list)),
        n_rl_datums=len(rl_list),
        n_opd_datums=len(opd_list),
        n_rl_tokens=n_rl_tok,
        n_opd_tokens=n_opd_tok,
        rl_loss_proxy=_extract_loss(rl_result),
        opd_nll=_extract_loss(opd_result),
        lambda_opd=requested_lambda,
        projection_coverage=float(projection_coverage),
        reject_rate=float(reject_rate),
        policy_version=policy_version,
        n_rl_forward_backward=n_rl_fb,
        n_opd_forward_backward=n_opd_fb,
        n_optimizer_steps=1,
        opd_to_rl_token_ratio=(n_opd_tok / n_rl_tok) if n_rl_tok else float(n_opd_tok),
        rl_opd_exact_target_overlap_rate=float(overlap_rate),
    )


@dataclass
class HybridLoopState:
    """Tracks the sync on-policy cycle: rollout version must equal last update."""

    policy_version: str
    global_optimizer_step: int = 0
    global_rollout_batch: int = 0
    call_log: list[str] = field(default_factory=list)

    def bump_after_update(self) -> str:
        try:
            n = int("".join(ch for ch in self.policy_version if ch.isdigit()) or "0")
        except ValueError:
            n = 0
        self.policy_version = f"v{n + 1}"
        return self.policy_version


async def run_hybrid_training_step(
    *,
    training_client: Any,
    groups: Sequence[HybridRolloutGroup],
    rl_datums_by_query: dict[str, list[Any]],
    policy_version: str,
    lambda_opd: float,
    component_id: str,
    teacher_event_fn: TeacherEventFn | None,
    rl_loss_fn: str = "cispo",
    rl_loss_fn_config: dict[str, Any] | None = None,
    adam_params: Any = None,
    num_substeps: int = 4,
    encode_fn: EncodeFn | None = None,
    opd_states_per_trajectory: int = 3,
    seed: int = 0,
    loop_state: HybridLoopState | None = None,
    opd_loss: str = "sr_opd_ce",
    opd_gate_beta: float = SCAPE_RL_OPD_GATE_BETA,
) -> list[HybridStepMetrics]:
    """One rollout batch → matched hybrid substeps → policy version bump."""
    batch = prepare_hybrid_batch(
        groups=groups,
        rl_datums_by_query=rl_datums_by_query,
        policy_version=policy_version,
        lambda_opd=lambda_opd,
        component_id=component_id,
        teacher_event_fn=teacher_event_fn,
        encode_fn=encode_fn,
        opd_states_per_trajectory=opd_states_per_trajectory,
        seed=seed,
        opd_loss=opd_loss,
        opd_gate_beta=opd_gate_beta,
    )
    metrics_out: list[HybridStepMetrics] = []
    pairs = split_hybrid_substeps(batch.rl_datums, batch.opd_datums, num_substeps=num_substeps)
    for rl_sub, opd_sub in pairs:
        if not rl_sub and not opd_sub:
            continue
        metrics = await hybrid_train_substep(
            training_client=training_client,
            rl_datums=rl_sub,
            opd_datums=opd_sub,
            rl_loss_fn=rl_loss_fn,
            rl_loss_fn_config=rl_loss_fn_config,
            lambda_opd=lambda_opd,
            adam_params=adam_params or {},
            policy_version=policy_version,
            projection_coverage=float(batch.projection_stats.get("projection_coverage") or 0.0),
            reject_rate=float(batch.projection_stats.get("reject_rate") or 0.0),
            overlap_rate=float(batch.projection_stats.get("rl_opd_exact_target_overlap_rate") or 0.0),
            opd_loss=opd_loss,
        )
        metrics_out.append(metrics)
        if loop_state is not None:
            loop_state.global_optimizer_step += metrics.n_optimizer_steps
    if loop_state is not None:
        loop_state.global_rollout_batch += 1
        loop_state.bump_after_update()
    return metrics_out
