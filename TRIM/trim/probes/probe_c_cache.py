"""Probe C0 funnel and C1 frozen-Teacher cache.

One real collection, many components: each state is pushed through a real
Teacher side-branch execution, a DIRECT projection, native encoding and a
frozen-Teacher logp cache. Only *real capability* sources count; mask flips,
render-string changes and synthetic events are not component effects.

The cache stores selected-target token logp (CPU fp32) and provenance only —
never full-vocabulary logits, never current-Student logp. Any model/branch/
encoding change invalidates a key loudly (recompute), never a silent hit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from trim.probes.probe_c_contract import HarnessPairSpec
from trim.probes.probe_c_data import ProbeCState
from trim.state.snapshot import stable_hash

# ---------------------------------------------------------------------------
# C0 funnel
# ---------------------------------------------------------------------------

STAGE_OPPORTUNITY = "opportunity"
STAGE_TRIGGERED = "triggered"
STAGE_REAL_SOURCE = "real_capability_source"
STAGE_EVENT_EMITTED = "event_emitted"
STAGE_PROJECTION_DIRECT = "projection_direct"
STAGE_REALIZABLE = "realizable"
STAGE_ENCODING_VALID = "encoding_valid"
STAGE_NONEMPTY_MASK = "nonempty_supervised_mask"
STAGE_GRADIENT_ROW = "gradient_bearing_row"

FUNNEL_STAGES = (
    STAGE_OPPORTUNITY,
    STAGE_TRIGGERED,
    STAGE_REAL_SOURCE,
    STAGE_EVENT_EMITTED,
    STAGE_PROJECTION_DIRECT,
    STAGE_REALIZABLE,
    STAGE_ENCODING_VALID,
    STAGE_NONEMPTY_MASK,
    STAGE_GRADIENT_ROW,
)

STATUS_UNREGISTERED = "UNREGISTERED"
STATUS_NO_TRIGGER = "NO_TRIGGER"
STATUS_NO_SUPPORT = "NO_SUPPORT"
STATUS_INVALID_MEASUREMENT = "INVALID_MEASUREMENT"
STATUS_READY = "READY"
STATUSES = (
    STATUS_UNREGISTERED,
    STATUS_NO_TRIGGER,
    STATUS_NO_SUPPORT,
    STATUS_INVALID_MEASUREMENT,
    STATUS_READY,
)

# v4 funnel schema (measurement_v4): attempted vs passed are explicit, so a
# row that *reaches* the DIRECT check and fails there is never counted as a
# DIRECT pass (the legacy ``stage_reached >= idx`` semantics conflated them).
FUNNEL_SCHEMA_VERSION_V4 = "probe_c_funnel_v4"

# New-stage name for supervision rows: cache-valid non-empty does not imply a
# real nonzero gradient; the trainer records that separately.
STAGE_SUPERVISION_ROW = "supervision_row_ready"


class InfraFailure(RuntimeError):
    """Network/resource-class failure. Never auto-falls back to a fake path."""


@dataclass
class FunnelRow:
    """One attempt. Rejections are recorded too, not only successes."""

    state_id: str
    query_id: str
    component_id: str
    stage_reached: str
    status: str
    reject_reason: str = ""
    n_supervised_tokens: int = 0
    n_target_tokens: int = 0
    projection_kind: str = ""
    source_type: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_v4(self) -> "FunnelRowV4":
        """Legacy row → v4: ``stage_reached`` marks the furthest *attempted*
        stage (the furthest check the row entered), not a pass."""
        idx = FUNNEL_STAGES.index(self.stage_reached) if self.stage_reached in FUNNEL_STAGES else 0
        attempted = list(FUNNEL_STAGES[: idx + 1])
        passed = []  # legacy rows cannot recover pass semantics; see below
        failed = self.stage_reached if self.status != STATUS_READY else None
        reasons = [self.reject_reason] if self.reject_reason else []
        if self.status == STATUS_READY:
            # Legacy READY rows passed every stage up to gradient row.
            passed = list(FUNNEL_STAGES)
            failed = None
        return FunnelRowV4(
            state_id=self.state_id,
            query_id=self.query_id,
            coalition_id=self.component_id,
            member_id=None,
            channel="legacy_migrated",
            attempted_stages=attempted,
            passed_stages=passed,
            failed_stage=failed,
            reason_codes=reasons,
            projection_kind=self.projection_kind or None,
            n_materialized_steps=0,
            n_supervised_tokens=self.n_supervised_tokens,
            detail={
                **self.detail,
                "legacy_status": self.status,
                "legacy_stage_reached": self.stage_reached,
                "legacy_source_type": self.source_type,
                "n_target_tokens": self.n_target_tokens,
            },
        )


@dataclass
class FunnelRowV4:
    """One attempt with explicit attempted/passed stage lists (measurement_v4).

    ``passed_stages`` only ever contains stages whose *check actually passed*;
    ``failed_stage`` + ``reason_codes`` carry the per-stage rejection cause.
    Legacy ``stage_reached`` semantics (furthest stage *entered*) cannot
    distinguish a DIRECT pass from a DIRECT-check failure and is superseded.
    """

    state_id: str
    query_id: str
    coalition_id: str
    member_id: str | None
    channel: str  # native_event_action | transform_then_policy_action | legacy_migrated
    attempted_stages: list[str] = field(default_factory=list)
    passed_stages: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    projection_kind: str | None = None
    n_materialized_steps: int = 0
    n_supervised_tokens: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def mark_attempt(self, stage: str) -> None:
        if stage not in self.attempted_stages:
            self.attempted_stages.append(stage)

    def mark_pass(self, stage: str) -> None:
        self.mark_attempt(stage)
        if stage not in self.passed_stages:
            self.passed_stages.append(stage)

    def fail(self, stage: str, reasons: Sequence[str]) -> None:
        self.mark_attempt(stage)
        self.failed_stage = stage
        for r in reasons:
            if r and r not in self.reason_codes:
                self.reason_codes.append(str(r))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FUNNEL_SCHEMA_VERSION_V4,
            **asdict(self),
        }


@dataclass
class FunnelCounts:
    """Raw counters. Zero denominators report N/A, never ``max(1, n)``."""

    component_id: str
    rows: list[FunnelRow] = field(default_factory=list)

    def count(self, stage: str) -> int:
        idx = FUNNEL_STAGES.index(stage)
        return sum(1 for r in self.rows if FUNNEL_STAGES.index(r.stage_reached) >= idx)

    # -- v4 (attempted/passed) ------------------------------------------
    def rows_v4(self) -> list["FunnelRowV4"]:
        return [r.to_v4() for r in self.rows]

    def count_attempted_v4(self, stage: str) -> int:
        return sum(1 for r in self.rows_v4() if stage in r.attempted_stages)

    def count_pass_v4(self, stage: str) -> int:
        """Only explicit passes count; a failed DIRECT check is never a DIRECT pass."""
        return sum(1 for r in self.rows_v4() if stage in r.passed_stages)

    def failure_reason_counts_v4(self, stage: str | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows_v4():
            if stage is not None and r.failed_stage != stage:
                continue
            for rc in r.reason_codes:
                out[rc] = out.get(rc, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_dict_v4(self) -> dict[str, Any]:
        per_stage: dict[str, dict[str, int]] = {}
        for stage in FUNNEL_STAGES:
            per_stage[stage] = {
                "attempted": self.count_attempted_v4(stage),
                "passed": self.count_pass_v4(stage),
                "failed_here": sum(
                    1 for r in self.rows_v4() if r.failed_stage == stage
                ),
            }
        rows = self.rows_v4()
        n_supervised = sum(1 for r in rows if r.n_supervised_tokens > 0)
        n_states = len(rows)
        n_pass_supervision = self.count_pass_v4(STAGE_SUPERVISION_ROW) or sum(
            1 for r in rows if STAGE_GRADIENT_ROW in r.passed_stages
        )
        return {
            "schema_version": FUNNEL_SCHEMA_VERSION_V4,
            "component_id": self.component_id,
            "per_stage": per_stage,
            "state_level": {
                "n_states": n_states,
                "n_supervision_rows_ready": n_pass_supervision,
                "supervision_rows_over_states": self._ratio(n_pass_supervision, n_states),
                "n_states_with_supervised_tokens": n_supervised,
                "n_queries": len({r.query_id for r in rows}),
                "n_covered_queries": len(
                    {r.query_id for r in rows if r.n_supervised_tokens > 0}
                ),
                "n_supervised_tokens": sum(r.n_supervised_tokens for r in rows),
                "denominators_note": "state-level counts; one state may carry "
                "multiple targets — target-level stats need target rows, not "
                "state rows",
            },
            "failure_reason_counts": self.failure_reason_counts_v4(),
        }

    def _ratio(self, num: int, denom: int) -> Any:
        return "N/A" if denom == 0 else num / denom

    def ratios(self) -> dict[str, Any]:
        n_all = len(self.rows)
        n_opp = self.count(STAGE_OPPORTUNITY)
        n_trig = self.count(STAGE_TRIGGERED)
        n_direct = self.count(STAGE_PROJECTION_DIRECT)
        n_valid = self.count(STAGE_ENCODING_VALID)
        n_supervised = sum(1 for r in self.rows if r.n_supervised_tokens > 0)
        n_queries = len({r.query_id for r in self.rows})
        covered = len({r.query_id for r in self.rows if r.n_supervised_tokens > 0})
        return {
            "trigger_over_opportunity": self._ratio(n_trig, n_opp),
            "direct_over_trigger": self._ratio(n_direct, n_trig),
            "valid_over_attempt": self._ratio(n_valid, n_all),
            "supervised_states_over_all_states": self._ratio(n_supervised, n_all),
            "unique_covered_queries_over_all_queries": self._ratio(covered, n_queries),
            "n_states": n_all,
            "n_queries": n_queries,
            "n_covered_queries": covered,
            "n_supervised_tokens": sum(r.n_supervised_tokens for r in self.rows),
            "note": "legacy stage_reached>=idx semantics: counts rows that "
            "REACHED a stage (including rows that failed its check). Use "
            "to_dict_v4() for attempted/passed semantics",
        }

    def status(self, *, min_covered_queries: int = 16, min_supervised_tokens: int = 1) -> str:
        if any(r.status == STATUS_INVALID_MEASUREMENT for r in self.rows):
            # A measurement bug poisons the ranking; fix and re-run.
            if all(r.status == STATUS_INVALID_MEASUREMENT for r in self.rows):
                return STATUS_INVALID_MEASUREMENT
        statuses = {r.status for r in self.rows}
        if STATUS_UNREGISTERED in statuses and STATUS_READY not in statuses:
            return STATUS_UNREGISTERED
        if self.count(STAGE_GRADIENT_ROW) == 0:
            if self.count(STAGE_TRIGGERED) == 0:
                return STATUS_NO_TRIGGER
            return STATUS_NO_SUPPORT
        supervised_tokens = sum(r.n_supervised_tokens for r in self.rows)
        covered = len({r.query_id for r in self.rows if r.n_supervised_tokens > 0})
        if supervised_tokens < min_supervised_tokens or covered < min_covered_queries:
            return STATUS_NO_SUPPORT
        return STATUS_READY

    def support_ready(self, *, min_covered_queries: int = 16, min_supervised_tokens: int = 1) -> bool:
        return self.status(
            min_covered_queries=min_covered_queries,
            min_supervised_tokens=min_supervised_tokens,
        ) == STATUS_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "status": self.status(),
            "ratios": self.ratios(),
            "n_rows": len(self.rows),
            "stage_counts": {s: self.count(s) for s in FUNNEL_STAGES},
            "v4": self.to_dict_v4(),
        }


def funnel_status(
    rows: Sequence[FunnelRow],
    *,
    component_id: str,
    min_covered_queries: int = 16,
    min_supervised_tokens: int = 1,
) -> str:
    return FunnelCounts(component_id=component_id, rows=list(rows)).status(
        min_covered_queries=min_covered_queries,
        min_supervised_tokens=min_supervised_tokens,
    )


# ---------------------------------------------------------------------------
# frozen Teacher cache
# ---------------------------------------------------------------------------


def compute_cache_key(
    *,
    teacher_policy_checksum: str,
    model_revision: str,
    tokenizer_hash: str,
    chat_template_hash: str,
    env_content_hash: str,
    component_revision: str,
    resolved_pair_hash: str,
    teacher_prompt_ids_hash: str,
    target_action_ids_hash: str,
    effective_schema_hash: str,
    truncation_policy_hash: str,
    enable_thinking: bool,
) -> str:
    """Minimal cache key (§7.2). A miss means recompute — never a silent fallback."""
    return stable_hash(
        {
            "teacher_policy_checksum": teacher_policy_checksum,
            "model_revision": model_revision,
            "tokenizer_hash": tokenizer_hash,
            "chat_template_hash": chat_template_hash,
            "env_content_hash": env_content_hash,
            "component_revision": component_revision,
            "resolved_pair_hash": resolved_pair_hash,
            "teacher_prompt_ids_hash": teacher_prompt_ids_hash,
            "target_action_ids_hash": target_action_ids_hash,
            "effective_schema_hash": effective_schema_hash,
            "truncation_policy_hash": truncation_policy_hash,
            "enable_thinking": bool(enable_thinking),
        }
    )


@dataclass
class TeacherTarget:
    """Cached frozen-Teacher supervision for one state."""

    cache_key: str
    state_id: str
    query_id: str
    component_id: str
    target_action: dict[str, Any]
    target_action_ids: list[int]
    # Selected-target token logp under the frozen Teacher (CPU fp32).
    teacher_token_logprobs: list[float]
    # Effective supervision weights (gate is *not* folded into these).
    weights: list[float]
    student_prompt_ids_hash: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.target_action_ids)
        if len(self.teacher_token_logprobs) != n:
            raise ValueError(
                f"target token length mismatch: ids={n} "
                f"teacher_logprobs={len(self.teacher_token_logprobs)}"
            )
        if self.weights and len(self.weights) != n:
            raise ValueError(
                f"weight length mismatch: ids={n} weights={len(self.weights)}"
            )
        if not self.weights:
            self.weights = [1.0] * n

    @property
    def n_target_tokens(self) -> int:
        return len(self.target_action_ids)


class TeacherLogpFn(Protocol):
    """Frozen-theta0 teacher scoring: prompt ids + target ids -> token logp."""

    def __call__(
        self, prompt_ids: Sequence[int], target_ids: Sequence[int]
    ) -> list[float]: ...


class FrozenTeacherCache:
    """Immutable-per-key Teacher cache. Student updates never invalidate it."""

    def __init__(self) -> None:
        self._rows: dict[str, TeacherTarget] = {}
        self.hits = 0
        self.misses = 0
        self.invalidations: list[dict[str, Any]] = []
        self.compute_seconds = 0.0

    def get(self, key: str) -> TeacherTarget | None:
        row = self._rows.get(key)
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row

    def put(self, row: TeacherTarget) -> None:
        self._rows[row.cache_key] = row

    def compute_or_fetch(
        self,
        key: str,
        *,
        recompute: Callable[[], TeacherTarget],
    ) -> TeacherTarget:
        """Cache miss recomputes. Model/branch/encoding changes never hit."""
        row = self.get(key)
        if row is not None:
            return row
        start = time.time()
        row = recompute()
        self.compute_seconds += time.time() - start
        if row.cache_key != key:
            raise ValueError(
                f"recompute produced key mismatch: want={key} got={row.cache_key}"
            )
        self.put(row)
        return row

    def invalidate(self, reason: str, *, detail: Mapping[str, Any] | None = None) -> int:
        n = len(self._rows)
        self._rows.clear()
        self.invalidations.append(
            {"reason": str(reason), "n_dropped": n, "detail": dict(detail or {})}
        )
        return n

    def __len__(self) -> int:
        return len(self._rows)

    def manifest(self) -> dict[str, Any]:
        """TEACHER_CACHE_MANIFEST.json content: keys / bytes / compute cost."""
        keys = sorted(self._rows)
        blob = json.dumps(keys).encode("utf-8")
        return {
            "n_entries": len(keys),
            "cache_keys": keys,
            "approx_key_bytes": len(blob),
            "approx_value_bytes": sum(
                4 * (r.n_target_tokens * 2) for r in self._rows.values()
            ),
            "hits": self.hits,
            "misses": self.misses,
            "compute_seconds": round(self.compute_seconds, 3),
            "invalidations": list(self.invalidations),
            "stores_full_vocab_logits": False,
            "stores_student_logp": False,
        }

    def write_manifest(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "TEACHER_CACHE_MANIFEST.json"
        path.write_text(
            json.dumps(self.manifest(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# real teacher cache construction (C0)
# ---------------------------------------------------------------------------


class TeacherBranchExecutor(Protocol):
    """Runs the real component side branch on a forked student snapshot."""

    def __call__(
        self, state: ProbeCState, pair: HarnessPairSpec
    ) -> "TeacherBranchOutcome": ...


@dataclass
class TeacherBranchOutcome:
    triggered: bool
    source_type: str  # capability_effect | synthetic_heuristic | skip_* ...
    events: list[Any] = field(default_factory=list)
    projection_kind: str = ""
    projected_steps: list[Any] = field(default_factory=list)
    reject_reason: str = ""
    realizable: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


def default_teacher_executor(
    component_id: str,
    *,
    teacher_kind: str = "upstream",
) -> TeacherBranchExecutor:
    """Real side-branch executor: TEACHER_REGISTRY branch + DIRECT projection.

    ``component_id`` may be a single id or an explicit coalition (comma list /
    sequence semantics via ``component_ids_of``). Per-member source types are
    recorded in ``detail["members"]``; unregistered members never fabricate
    supervision and never silently fall back to a synthetic heuristic.
    """
    from trim.training.opd_dataset import ProjectionAudit, project_and_materialize
    from trim.training.rl_opd_types import StudentDecisionPoint
    from trim.training.teacher_branch import (
        SOURCE_CAPABILITY,
        SOURCE_SKIP_UNREGISTERED,
        SOURCE_SKIP_UNTRIGGERED,
        SOURCE_SYNTHETIC,
        coerce_teacher_branch,
    )

    def _run(state: ProbeCState, pair: HarnessPairSpec) -> TeacherBranchOutcome:
        from trim.state.snapshot import EnvironmentSnapshot
        from trim.training.four_cell_runtime import component_ids_of, teacher_for

        snap = EnvironmentSnapshot.from_dict(state.student_snapshot)
        point = StudentDecisionPoint(
            episode_id=f"{state.query_id}:{state.rollout_seed}",
            query_id=state.query_id,
            rollout_idx=int(state.rollout_seed),
            turn_id=int(state.turn_id),
            policy_version=frozen_policy_version(state),
            pre_action_snapshot=snap,
            pre_action_snapshot_hash=snap.content_hash(),
            student_model_input="",
            student_action_tokens=list(state.student_action_ids),
            student_action_text="",
            action_tool_names=[],
            structurally_valid=True,
            reward=0.0,
        )
        members = component_ids_of(component_id) or [str(component_id)]
        real_events: list[Any] = []
        member_rows: list[dict[str, Any]] = []
        any_triggered = False
        any_synthetic = False
        for cid in members:
            fn = teacher_for(cid, teacher_kind=teacher_kind)
            raw = fn(point) if fn is not None else []
            branch = coerce_teacher_branch(raw, point=point, component_id=cid)
            src = str(branch.source_type or "")
            member_rows.append(
                {
                    "component_id": cid,
                    "source_type": src,
                    "triggered": bool(branch.triggered),
                    "skip_reason": branch.skip_reason,
                }
            )
            if src == SOURCE_SKIP_UNREGISTERED or (
                branch.skip_reason and "unregistered" in str(branch.skip_reason)
            ):
                continue
            if src == SOURCE_SYNTHETIC:
                any_synthetic = True
                continue
            if not branch.triggered:
                continue
            any_triggered = True
            real_events.extend(branch.events)

        detail = {"members": member_rows}
        registered = [r for r in member_rows if r["source_type"] not in {
            SOURCE_SKIP_UNREGISTERED,
        }]
        if not registered:
            return TeacherBranchOutcome(
                triggered=False,
                source_type=SOURCE_SKIP_UNREGISTERED,
                events=[],
                reject_reason="no_real_teacher_implementation",
                detail=detail,
            )
        if not any_triggered:
            return TeacherBranchOutcome(
                triggered=False,
                source_type=SOURCE_SKIP_UNTRIGGERED
                if not any_synthetic
                else SOURCE_SYNTHETIC,
                events=[],
                reject_reason="synthetic_heuristic_not_a_real_capability_source"
                if any_synthetic
                else "not_triggered",
                detail=detail,
            )
        if not real_events:
            return TeacherBranchOutcome(
                triggered=True,
                source_type=SOURCE_SYNTHETIC if any_synthetic else SOURCE_CAPABILITY,
                events=[],
                reject_reason="no_real_capability_events"
                if not any_synthetic
                else "synthetic_heuristic_not_a_real_capability_source",
                detail=detail,
            )
        audit = ProjectionAudit(component_id=str(component_id))
        projection, steps = project_and_materialize(
            student_snapshot=snap,
            teacher_events=list(real_events),
            student_mask=dict(pair.student_mask),
            component_id=str(component_id),
            audit=audit,
        )
        outcome = TeacherBranchOutcome(
            triggered=True,
            source_type=SOURCE_CAPABILITY,
            events=list(real_events),
            projection_kind=str(getattr(projection.kind, "value", projection.kind)),
            projected_steps=list(steps),
            reject_reason=str(getattr(projection, "reject_reason", "") or ""),
            realizable=bool(steps),
            detail=detail,
        )
        outcome.detail["projection_audit"] = audit.to_dict()
        return outcome

    return _run


def frozen_policy_version(state: ProbeCState) -> str:
    return f"theta0:{state.policy_checksum[:12]}"


@dataclass
class CachedSupervisionRow:
    state: ProbeCState
    target: TeacherTarget
    funnel: FunnelRow | None = None


def build_real_teacher_cache(
    states: Sequence[ProbeCState],
    pair: HarnessPairSpec,
    *,
    component_id: str,
    executor: TeacherBranchExecutor,
    encoder: Callable[[ProbeCState, Mapping[str, Any]], tuple[list[int], list[bool]]],
    teacher_logp_fn: TeacherLogpFn,
    cache: FrozenTeacherCache | None = None,
    token_weights: Callable[[Sequence[bool]], list[float]] | None = None,
    teacher_prompt_fn: Callable[[ProbeCState, "TeacherBranchOutcome"], Sequence[int]]
    | None = None,
) -> tuple[list[CachedSupervisionRow], FunnelCounts]:
    """C0+C1 data path: real branch → DIRECT projection → native encoding → cache.

    ``encoder(state, target_action)`` must return native completion token ids
    and a supervision mask for the target action under the Student view (H0).
    ``teacher_logp_fn`` scores those tokens under the frozen Teacher (theta0).
    ``teacher_prompt_fn`` must return the *Teacher-side* prompt ids (the H_m
    branch view); without it the Student prompt ids are used, which is only
    correct when m does not change the context view.
    """
    from trim.training.action_encoding import prompt_ids_hash

    cache = cache if cache is not None else FrozenTeacherCache()
    rows: list[CachedSupervisionRow] = []
    funnel = FunnelCounts(component_id=component_id)
    pair_hash = pair.resolved_pair_hash()

    for state in states:
        fid = state.state_id
        # Stage 1: opportunity (the state exists under H0 with a legal history)
        funnel.rows.append(
            FunnelRow(
                state_id=fid,
                query_id=state.query_id,
                component_id=component_id,
                stage_reached=STAGE_OPPORTUNITY,
                status=STATUS_NO_TRIGGER,
            )
        )
        row = funnel.rows[-1]
        try:
            outcome = executor(state, pair)
        except InfraFailure:
            raise
        except Exception as exc:  # measurement/infra — never a silent fallback
            row.stage_reached = STAGE_OPPORTUNITY
            row.status = STATUS_INVALID_MEASUREMENT
            row.reject_reason = f"executor_error:{type(exc).__name__}:{exc}"
            continue
        row.detail = dict(getattr(outcome, "detail", None) or {})

        if not outcome.triggered:
            # Reached the opportunity stage only; the trigger never fired.
            row.stage_reached = STAGE_OPPORTUNITY
            row.status = (
                STATUS_UNREGISTERED
                if outcome.source_type.endswith("unregistered")
                else STATUS_NO_TRIGGER
            )
            row.reject_reason = outcome.reject_reason or "not_triggered"
            row.source_type = outcome.source_type
            continue
        row.stage_reached = STAGE_TRIGGERED

        if outcome.source_type != "capability_effect":
            row.stage_reached = STAGE_REAL_SOURCE
            row.status = STATUS_NO_SUPPORT
            row.reject_reason = outcome.reject_reason or f"source={outcome.source_type}"
            row.source_type = outcome.source_type
            continue
        row.stage_reached = STAGE_REAL_SOURCE
        row.source_type = outcome.source_type

        if not outcome.events:
            row.stage_reached = STAGE_REAL_SOURCE
            row.status = STATUS_NO_SUPPORT
            row.reject_reason = "no_event_emitted"
            continue
        row.stage_reached = STAGE_EVENT_EMITTED

        if outcome.projection_kind != "direct" or not outcome.projected_steps:
            row.stage_reached = STAGE_PROJECTION_DIRECT
            row.status = STATUS_NO_SUPPORT
            row.projection_kind = outcome.projection_kind
            row.reject_reason = outcome.reject_reason or (
                f"projection_kind={outcome.projection_kind or 'none'}"
            )
            continue
        row.stage_reached = STAGE_PROJECTION_DIRECT
        row.projection_kind = outcome.projection_kind

        if not outcome.realizable:
            row.stage_reached = STAGE_REALIZABLE
            row.status = STATUS_NO_SUPPORT
            row.reject_reason = outcome.reject_reason or "unrealizable"
            continue
        row.stage_reached = STAGE_REALIZABLE

        step = outcome.projected_steps[0]
        target_action = dict(step.target_action)
        try:
            target_ids, sup_mask = encoder(state, target_action)
        except Exception as exc:
            row.stage_reached = STAGE_REALIZABLE
            row.status = STATUS_INVALID_MEASUREMENT
            row.reject_reason = f"encoding_error:{type(exc).__name__}:{exc}"
            continue
        if not target_ids:
            row.stage_reached = STAGE_ENCODING_VALID
            row.status = STATUS_INVALID_MEASUREMENT
            row.reject_reason = "empty_target_encoding"
            continue
        row.stage_reached = STAGE_ENCODING_VALID
        row.n_target_tokens = len(target_ids)

        if not any(sup_mask):
            row.stage_reached = STAGE_NONEMPTY_MASK
            row.status = STATUS_NO_SUPPORT
            row.reject_reason = "empty_supervised_mask"
            continue
        row.stage_reached = STAGE_NONEMPTY_MASK

        weights = (
            token_weights(sup_mask)
            if token_weights is not None
            else [1.0 if m else 0.0 for m in sup_mask]
        )
        if teacher_prompt_fn is not None:
            teacher_prompt_ids = list(teacher_prompt_fn(state, outcome))
        else:
            teacher_prompt_ids = list(state.effective_student_prompt_ids)
        if not teacher_prompt_ids:
            row.stage_reached = STAGE_ENCODING_VALID
            row.status = STATUS_INVALID_MEASUREMENT
            row.reject_reason = "empty_teacher_prompt_ids"
            continue
        key = compute_cache_key(
            teacher_policy_checksum=state.policy_checksum,
            model_revision=state.model_revision,
            tokenizer_hash=state.tokenizer_hash,
            chat_template_hash=state.chat_template_hash,
            env_content_hash=state.env_content_hash,
            component_revision=pair.component_revision,
            resolved_pair_hash=pair_hash,
            teacher_prompt_ids_hash=prompt_ids_hash(teacher_prompt_ids),
            target_action_ids_hash=stable_hash(list(target_ids)),
            effective_schema_hash=stable_hash(state.effective_tool_schema),
            truncation_policy_hash=state.truncation_policy_hash,
            enable_thinking=state.enable_thinking,
        )

        def _recompute(
            _state: ProbeCState = state,
            _ids: list[int] = list(target_ids),
            _action: dict[str, Any] = dict(target_action),
            _weights: list[float] = list(weights),
            _key: str = key,
            _prompt_ids: list[int] = list(teacher_prompt_ids),
            _pid_hash: str = prompt_ids_hash(teacher_prompt_ids),
        ) -> TeacherTarget:
            logp = list(teacher_logp_fn(_prompt_ids, _ids))
            return TeacherTarget(
                cache_key=_key,
                state_id=_state.state_id,
                query_id=_state.query_id,
                component_id=component_id,
                target_action=_action,
                target_action_ids=_ids,
                teacher_token_logprobs=[float(x) for x in logp],
                weights=list(_weights),
                student_prompt_ids_hash=_pid_hash,
                provenance={
                    "source_type": "capability_effect",
                    "projection_kind": "direct",
                    "pair_hash": pair_hash,
                    "env_content_hash": _state.env_content_hash,
                    "policy_checksum": _state.policy_checksum,
                    "teacher_prompt_ids_hash": _pid_hash,
                    "teacher_prompt_view": "h_m_branch_view"
                    if teacher_prompt_fn is not None
                    else "student_prompt_fallback",
                },
            )

        try:
            target = cache.compute_or_fetch(key, recompute=_recompute)
        except InfraFailure:
            raise
        except ValueError as exc:
            row.stage_reached = STAGE_ENCODING_VALID
            row.status = STATUS_INVALID_MEASUREMENT
            row.reject_reason = f"cache_error:{exc}"
            continue

        row.stage_reached = STAGE_GRADIENT_ROW
        row.status = STATUS_READY
        row.n_supervised_tokens = sum(1 for w in target.weights if w > 0)
        rows.append(
            CachedSupervisionRow(state=state, target=target, funnel=row)
        )
    return rows, funnel


def write_funnel_jsonl(funnel: FunnelCounts, out_dir: str | Path) -> Path:
    """TEACHER_FUNNEL.jsonl: all attempts and reject reasons, not only successes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "TEACHER_FUNNEL.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in funnel.rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


__all__ = [
    "CachedSupervisionRow",
    "FrozenTeacherCache",
    "FUNNEL_STAGES",
    "FUNNEL_SCHEMA_VERSION_V4",
    "FunnelCounts",
    "FunnelRow",
    "FunnelRowV4",
    "InfraFailure",
    "STAGE_SUPERVISION_ROW",
    "STATUS_INVALID_MEASUREMENT",
    "STATUS_NO_SUPPORT",
    "STATUS_NO_TRIGGER",
    "STATUS_READY",
    "STATUS_UNREGISTERED",
    "STATUSES",
    "TeacherBranchExecutor",
    "TeacherBranchOutcome",
    "TeacherLogpFn",
    "TeacherTarget",
    "build_real_teacher_cache",
    "compute_cache_key",
    "default_teacher_executor",
    "funnel_status",
    "write_funnel_jsonl",
]
