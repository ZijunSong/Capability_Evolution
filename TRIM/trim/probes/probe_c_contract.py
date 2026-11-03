"""Probe C explicit H0 / H0+m contract (GPU-free, hashable, unambiguous).

This module owns the *pair* semantics for Probe C. ``add_from_h0`` starts from
``zero_mask(harness)`` (H0: advanced components off) and turns on exactly the
requested component set in the Teacher mask; the Student stays on H0 for the
whole probe. ``legacy_full_minus`` preserves the historical Full-minus-m /
Full interpretation and must never be mixed into the same ranking column.

Nothing here infers masks from the old ``--component m`` semantics. Every
process must resolve and print the masks, the base contract hash and the
component revision before any expensive stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from trim.adapters.components import (
    all_component_ids,
    assert_mask_diff_only,
    clone_mask,
    full_mask,
    minus_mask,
    zero_mask,
)
from trim.state.snapshot import stable_hash

SCHEMA_VERSION = "probe_c_v3"

PAIR_MODE_ADD = "add_from_h0"
PAIR_MODE_LEGACY = "legacy_full_minus"
PAIR_MODES = (PAIR_MODE_ADD,PAIR_MODE_LEGACY)

STATE_SOURCE_REAL = "real_h0_rollout"
PROJECTION_DIRECT_ONLY = "direct_only"
PROBE_OBJECTIVE = "frozen_projected_gap"

# Placeholder that the audit stage must refuse beyond audit itself.
UNRESOLVED_REVISION = "RESOLVE_AND_FREEZE_BEFORE_RUN"

SPLIT_TRAIN = "C_train"
SPLIT_SELECT = "C_select"
SPLIT_AUDIT = "C_audit"
SPLIT_TASK_TEST = "Task_test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_SELECT, SPLIT_AUDIT, SPLIT_TASK_TEST)


class ContractError(ValueError):
    """Raised for any ambiguous or unfrozen contract input."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def base_contract_hash(
    *,
    harness: str | None = None,
    base_schema: Mapping[str, Any] | None = None,
) -> str:
    """Hash of the frozen H0 base contract (not of component configuration)."""
    payload = {
        "harness": harness or "Harness-1",
        "base_schema": dict(base_schema or {}),
        "note": "H0 keeps deterministic base execution guarantees; advanced zero "
        "does not disable base execution/safety/counting.",
    }
    return stable_hash(payload)


def component_revision_of(
    component_id: str,
    *,
    git_sha: str = "",
    file_hashes: Mapping[str, str] | None = None,
) -> str:
    """Revision string for one component implementation (must be explicit)."""
    payload = {
        "component_id": str(component_id),
        "git_sha": str(git_sha or ""),
        "file_hashes": {str(k): str(v) for k, v in dict(file_hashes or {}).items()},
    }
    if not payload["git_sha"] and not payload["file_hashes"]:
        raise ContractError(
            f"component_revision for {component_id!r} is empty; provide git_sha "
            "and/or file hashes — classification labels are not execution evidence"
        )
    return stable_hash(payload)


@dataclass(frozen=True)
class HarnessPairSpec:
    """Explicit student/teacher deployment pair for one Probe-C candidate."""

    pair_mode: str
    component_ids: tuple[str, ...]
    student_mask: dict[str, bool]
    teacher_mask: dict[str, bool]
    base_contract_hash: str
    component_revision: str
    harness: str = "Harness-1"
    # Components m depends on that are *inside* component_ids (explicit composite).
    declared_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.pair_mode not in PAIR_MODES:
            raise ContractError(f"unknown pair_mode={self.pair_mode!r}; allowed {PAIR_MODES}")
        if not self.component_ids:
            raise ContractError("HarnessPairSpec requires at least one component id")
        if not self.base_contract_hash:
            raise ContractError("base_contract_hash must be set")
        if not self.component_revision:
            raise ContractError("component_revision must be set")
        if self.component_revision == UNRESOLVED_REVISION:
            raise ContractError("component_revision is unresolved")
        self.validate()

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        student = self.student_mask
        teacher = self.teacher_mask
        if set(student) != set(teacher):
            raise ContractError(
                f"student/teacher mask key mismatch: "
                f"{sorted(set(student) ^ set(teacher))}"
            )
        if set(student) != set(all_component_ids(self.harness)):
            raise ContractError(
                "mask keys do not match harness taxonomy; "
                f"harness={self.harness!r}"
            )
        expected = set(self.component_ids)
        try:
            assert_mask_diff_only(student, teacher, expected_changed=expected)
        except AssertionError as exc:
            raise ContractError(str(exc)) from exc
        if self.pair_mode ==PAIR_MODE_ADD:
            want_student = zero_mask(self.harness)
            if student != want_student:
                raise ContractError(
                    "add_from_h0 requires student_mask == zero_mask(harness); "
                    "found non-zero advanced components"
                )
            for cid, on in teacher.items():
                if on != (cid in expected):
                    raise ContractError(
                        f"add_from_h0 teacher_mask must enable exactly "
                        f"{sorted(expected)}; {cid}={on}"
                    )
        else:  # legacy_full_minus: Full-minus-m student vs Full-with-m teacher
            want_student = minus_mask(
                self.component_ids[0], harness=self.harness
            ) if len(self.component_ids) == 1 else None
            if want_student is not None and student != want_student:
                raise ContractError(
                    "legacy_full_minus single-component student_mask must be "
                    "Full-minus-m (coalition_minus_mask)"
                )
            for cid in self.component_ids:
                if not teacher.get(cid, False):
                    raise ContractError(
                        f"legacy_full_minus teacher_mask must enable {cid}"
                    )
        missing_deps = [
            d for d in self.declared_dependencies if d not in expected
        ]
        if missing_deps:
            raise ContractError(
                f"declared_dependencies {missing_deps} must be listed inside "
                "component_ids (define m as an explicit small combination)"
            )

    # -- derived --------------------------------------------------------
    def diff_components(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                cid
                for cid in self.student_mask
                if self.student_mask[cid] != self.teacher_mask[cid]
            )
        )

    def resolved_pair_hash(self) -> str:
        return stable_hash(
            {
                "pair_mode": self.pair_mode,
                "component_ids": list(self.component_ids),
                "student_mask": dict(sorted(self.student_mask.items())),
                "teacher_mask": dict(sorted(self.teacher_mask.items())),
                "base_contract_hash": self.base_contract_hash,
                "component_revision": self.component_revision,
                "declared_dependencies": list(self.declared_dependencies),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["component_ids"] = list(self.component_ids)
        payload["declared_dependencies"] = list(self.declared_dependencies)
        payload["resolved_pair_hash"] = self.resolved_pair_hash()
        payload["diff_components"] = list(self.diff_components())
        return payload

    def describe(self) -> str:
        return (
            f"HarnessPairSpec(mode={self.pair_mode}, components={list(self.component_ids)}, "
            f"pair_hash={self.resolved_pair_hash()[:12]})"
        )


def make_add_pair(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
    base_contract: Mapping[str, Any] | None = None,
    git_sha: str = "",
    file_hashes: Mapping[str, str] | None = None,
    component_revision: str = "",
    declared_dependencies: Sequence[str] = (),
) -> HarnessPairSpec:
    """Build the explicit add pair: Student=H0, Teacher=H0+m.

    ``component_ids`` is the full explicit set the Teacher may enable; any
    required dependencies of m must be listed here (composite candidate).
    """
    ids = tuple(str(c).strip() for c in component_ids if str(c).strip())
    if not ids:
        raise ContractError("make_add_pair: empty component_ids")
    resolved = harness or "Harness-1"
    known = set(all_component_ids(resolved))
    unknown = [c for c in ids if c not in known]
    if unknown:
        raise ContractError(f"unknown component id(s) {unknown} for {resolved}")
    student = zero_mask(resolved)
    teacher = zero_mask(resolved)
    for cid in ids:
        teacher[cid] = True
    revision = component_revision or component_revision_of(
        ids[0], git_sha=git_sha, file_hashes=file_hashes
    )
    return HarnessPairSpec(
        pair_mode=PAIR_MODE_ADD,
        component_ids=ids,
        student_mask=student,
        teacher_mask=teacher,
        base_contract_hash=base_contract_hash(
            harness=resolved, base_schema=base_contract
        ),
        component_revision=revision,
        harness=resolved,
        declared_dependencies=tuple(str(d) for d in declared_dependencies),
    )


def make_legacy_full_minus_pair(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
    base_contract: Mapping[str, Any] | None = None,
    git_sha: str = "",
    file_hashes: Mapping[str, str] | None = None,
    component_revision: str = "",
) -> HarnessPairSpec:
    """Historical Full-minus-m student / Full+m teacher pair (legacy column)."""
    ids = tuple(str(c).strip() for c in component_ids if str(c).strip())
    if not ids:
        raise ContractError("make_legacy_full_minus_pair: empty component_ids")
    resolved = harness or "Harness-1"
    from trim.adapters.components import coalition_minus_mask

    student = coalition_minus_mask(ids, harness=resolved)
    teacher = full_mask(resolved)
    for cid in ids:
        teacher[cid] = True
    revision = component_revision or component_revision_of(
        ids[0], git_sha=git_sha, file_hashes=file_hashes
    )
    return HarnessPairSpec(
        pair_mode=PAIR_MODE_LEGACY,
        component_ids=ids,
        student_mask=student,
        teacher_mask=teacher,
        base_contract_hash=base_contract_hash(
            harness=resolved, base_schema=base_contract
        ),
        component_revision=revision,
        harness=resolved,
    )


# ---------------------------------------------------------------------------
# Probe-C experiment contract (config + manifest)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """Pilot split sizes. Freezing happens via ``frozen`` + manifest hash."""

    n_train: int = 128
    n_select: int = 64
    n_audit: int = 64
    states_per_query: int = 4
    group_key: str = "scenario_group_id"  # query / scenario grouping unit
    frozen: bool = False

    def max_states(self) -> dict[str, int]:
        k = int(self.states_per_query)
        return {
            SPLIT_TRAIN: int(self.n_train) * k,
            SPLIT_SELECT: int(self.n_select) * k,
            SPLIT_AUDIT: int(self.n_audit) * k,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def split_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class C1Budget:
    action_tokens_per_optimizer_step: int = 1024
    checkpoints: tuple[int, ...] = (0, 4, 8)
    borderline_max_step: int = 16
    loss_scale: float = 1.0
    gate_beta: float = 5.0
    learning_rate: float = 2e-5
    lora_rank: int = 16
    grad_clip: float = 1.0
    optimizer: str = "adamw"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoints"] = list(self.checkpoints)
        return payload


@dataclass(frozen=True)
class Thresholds:
    """Frozen pilot thresholds for C1 gates (design values, pre-registered)."""

    delta_nll: float = 0.0
    delta_dk: float = 0.0
    eps_name_js: float = 0.02
    eps_invalid: float = 0.02
    eps_j: float = 0.01
    min_covered_queries: int = 16
    min_supervised_tokens: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeCConfig:
    """Parsed ``probe_c_*.json`` contract. Kept free of private paths/keys."""

    schema_version: str
    model_id: str
    model_revision: str
    pair_mode: str
    components: list[str]
    state_source: str = STATE_SOURCE_REAL
    enable_thinking: bool = False
    teacher_policy_mode: str = "frozen_theta0"
    projection_mode: str = PROJECTION_DIRECT_ONLY
    probe_objective: str = PROBE_OBJECTIVE
    gate_beta: float = 5.0
    loss_scale: float = 1.0
    action_tokens_per_optimizer_step: int = 1024
    checkpoints: tuple[int, ...] = (0, 4, 8)
    borderline_max_step: int = 16
    train_seeds: tuple[int, ...] = (11, 23)
    candidate_limit: int = 8
    candidate_refresh_after_training: bool = False
    bootstrap_unit: str = "scenario_group"
    formal_harnessopd_method: str = "trim"
    formal_harnessopd_loss: str = "sr_opd_projected_gap"
    harness: str = "Harness-1"
    splits: SplitSpec = field(default_factory=SplitSpec)
    budget: C1Budget = field(default_factory=C1Budget)
    thresholds: Thresholds = field(default_factory=Thresholds)
    output_root: str = ""
    # Explicit small-combination candidates: candidate id -> member component
    # ids. Empty means each entry of ``components`` is a single component.
    coalition_members: dict[str, tuple[str, ...]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def members_of(self, candidate_id: str) -> tuple[str, ...]:
        return self.coalition_members.get(candidate_id) or (str(candidate_id),)

    # -- validation ----------------------------------------------------
    def validate(self, *, stage: str = "audit") -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if self.state_source != STATE_SOURCE_REAL:
            raise ContractError(
                f"state_source must be {STATE_SOURCE_REAL!r}; fake/smoke sources "
                "are forbidden for formal Probe C"
            )
        if self.projection_mode != PROJECTION_DIRECT_ONLY:
            raise ContractError("projection_mode must be 'direct_only' (first version)")
        if self.probe_objective != PROBE_OBJECTIVE:
            raise ContractError(
                f"probe_objective must be {PROBE_OBJECTIVE!r}; CE substitution is "
                "not allowed under the same name"
            )
        if self.pair_mode not in PAIR_MODES:
            raise ContractError(f"unknown pair_mode={self.pair_mode!r}")
        if not self.components:
            raise ContractError("components list is empty")
        if self.coalition_members:
            unknown = [c for c in self.coalition_members if c not in self.components]
            if unknown:
                raise ContractError(
                    f"coalition_members keys {unknown} not listed in components"
                )
            from trim.adapters.components import all_component_ids

            known = set(all_component_ids(self.harness))
            for key, members in self.coalition_members.items():
                if not members:
                    raise ContractError(f"coalition {key!r} has no members")
                bad = [m for m in members if m not in known]
                if bad:
                    raise ContractError(
                        f"coalition {key!r} has unknown members {bad}"
                    )
        if self.candidate_limit < 2:
            raise ContractError("candidate_limit must be >= 2")
        if float(self.loss_scale) != 1.0:
            raise ContractError(
                "C1 defines loss_scale=1; the formal joint lambda must not be "
                "copied into the probe micro-updates"
            )
        if not self.splits.frozen:
            raise ContractError("splits are not frozen; freeze SplitSpec before running")
        self.require_frozen_model(stage=stage)

    def require_frozen_model(self, *, stage: str) -> None:
        rev = str(self.model_revision or "").strip()
        if not rev or rev == UNRESOLVED_REVISION:
            raise ContractError(
                "model_revision is unresolved; resolve and freeze it before "
                f"stage {stage!r} (refusing to fake success and let the batch "
                "proceed to expensive stages)"
            )

    def budget_spec(self) -> C1Budget:
        return C1Budget(
            action_tokens_per_optimizer_step=int(self.action_tokens_per_optimizer_step),
            checkpoints=tuple(int(c) for c in self.checkpoints),
            borderline_max_step=int(self.borderline_max_step),
            loss_scale=float(self.loss_scale),
            gate_beta=float(self.gate_beta),
            learning_rate=float(self.budget.learning_rate),
            lora_rank=int(self.budget.lora_rank),
            grad_clip=float(self.budget.grad_clip),
            optimizer=str(self.budget.optimizer),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "pair_mode": self.pair_mode,
            "components": list(self.components),
            "state_source": self.state_source,
            "enable_thinking": self.enable_thinking,
            "teacher_policy_mode": self.teacher_policy_mode,
            "projection_mode": self.projection_mode,
            "probe_objective": self.probe_objective,
            "gate_beta": self.gate_beta,
            "loss_scale": self.loss_scale,
            "action_tokens_per_optimizer_step": self.action_tokens_per_optimizer_step,
            "checkpoints": list(self.checkpoints),
            "borderline_max_step": self.borderline_max_step,
            "train_seeds": list(self.train_seeds),
            "candidate_limit": self.candidate_limit,
            "candidate_refresh_after_training": self.candidate_refresh_after_training,
            "bootstrap_unit": self.bootstrap_unit,
            "formal_harnessopd_method": self.formal_harnessopd_method,
            "formal_harnessopd_loss": self.formal_harnessopd_loss,
            "harness": self.harness,
            "coalition_members": {
                k: list(v) for k, v in self.coalition_members.items()
            },
            "splits": self.splits.to_dict(),
            "budget": self.budget.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "output_root": self.output_root,
        }


def load_config(path: str | Path, *, stage: str = "audit") -> ProbeCConfig:
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"config not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError("config root must be a JSON object")
    splits_raw = dict(raw.get("splits") or {})
    budget_raw = dict(raw.get("budget") or {})
    thr_raw = dict(raw.get("thresholds") or {})
    cfg = ProbeCConfig(
        schema_version=str(raw.get("schema_version") or ""),
        model_id=str(raw.get("model_id") or ""),
        model_revision=str(raw.get("model_revision") or ""),
        pair_mode=str(raw.get("pair_mode") or ""),
        components=[str(c) for c in (raw.get("components") or [])],
        state_source=str(raw.get("state_source") or ""),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        teacher_policy_mode=str(raw.get("teacher_policy_mode") or "frozen_theta0"),
        projection_mode=str(raw.get("projection_mode") or ""),
        probe_objective=str(raw.get("probe_objective") or ""),
        gate_beta=float(raw.get("gate_beta", 5.0)),
        loss_scale=float(raw.get("loss_scale", 1.0)),
        action_tokens_per_optimizer_step=int(raw.get("action_tokens_per_optimizer_step", 1024)),
        checkpoints=tuple(int(c) for c in (raw.get("checkpoints") or (0, 4, 8))),
        borderline_max_step=int(raw.get("borderline_max_step", 16)),
        train_seeds=tuple(int(s) for s in (raw.get("train_seeds") or (11, 23))),
        candidate_limit=int(raw.get("candidate_limit", 8)),
        candidate_refresh_after_training=bool(
            raw.get("candidate_refresh_after_training", False)
        ),
        bootstrap_unit=str(raw.get("bootstrap_unit") or "scenario_group"),
        formal_harnessopd_method=str(raw.get("formal_harnessopd_method") or "trim"),
        formal_harnessopd_loss=str(raw.get("formal_harnessopd_loss") or "sr_opd_projected_gap"),
        harness=str(raw.get("harness") or "Harness-1"),
        splits=SplitSpec(
            n_train=int(splits_raw.get("n_train", 128)),
            n_select=int(splits_raw.get("n_select", 64)),
            n_audit=int(splits_raw.get("n_audit", 64)),
            states_per_query=int(splits_raw.get("states_per_query", 4)),
            group_key=str(splits_raw.get("group_key") or "scenario_group_id"),
            frozen=bool(splits_raw.get("frozen", False)),
        ),
        budget=C1Budget(
            action_tokens_per_optimizer_step=int(
                budget_raw.get("action_tokens_per_optimizer_step", 1024)
            ),
            checkpoints=tuple(int(c) for c in (budget_raw.get("checkpoints") or (0, 4, 8))),
            borderline_max_step=int(budget_raw.get("borderline_max_step", 16)),
            loss_scale=float(budget_raw.get("loss_scale", 1.0)),
            gate_beta=float(budget_raw.get("gate_beta", 5.0)),
            learning_rate=float(budget_raw.get("learning_rate", 2e-5)),
            lora_rank=int(budget_raw.get("lora_rank", 16)),
            grad_clip=float(budget_raw.get("grad_clip", 1.0)),
            optimizer=str(budget_raw.get("optimizer") or "adamw"),
        ),
        thresholds=Thresholds(
            delta_nll=float(thr_raw.get("delta_nll", 0.0)),
            delta_dk=float(thr_raw.get("delta_dk", 0.0)),
            eps_name_js=float(thr_raw.get("eps_name_js", 0.02)),
            eps_invalid=float(thr_raw.get("eps_invalid", 0.02)),
            eps_j=float(thr_raw.get("eps_j", 0.01)),
            min_covered_queries=int(thr_raw.get("min_covered_queries", 16)),
            min_supervised_tokens=int(thr_raw.get("min_supervised_tokens", 1)),
        ),
        output_root=str(raw.get("output_root") or ""),
        coalition_members={
            str(k): tuple(str(m) for m in v)
            for k, v in dict(raw.get("coalition_members") or {}).items()
        },
        raw=raw,
    )
    cfg.validate(stage=stage)
    return cfg


def build_manifest(
    config: ProbeCConfig,
    *,
    pairs: Mapping[str, HarnessPairSpec] | None = None,
    extra: Mapping[str, Any] | None = None,
    git_sha: str = "",
) -> dict[str, Any]:
    """MANIFEST.json content: revision / model / pair / objective / splits / budgets."""
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "pair_mode": config.pair_mode,
        "objective": {
            "probe_objective": config.probe_objective,
            "gate_beta": config.gate_beta,
            "loss_scale": config.loss_scale,
            "projection_mode": config.projection_mode,
            "teacher_policy_mode": config.teacher_policy_mode,
        },
        "formal_reference": {
            "method": config.formal_harnessopd_method,
            "loss": config.formal_harnessopd_loss,
            "note": "C1 is frozen-teacher fixed-state micro-updates; C2/C3 keep "
            "formal HarnessOPD semantics",
        },
        "splits": config.splits.to_dict(),
        "split_hash": config.splits.split_hash(),
        "budget": config.budget_spec().to_dict(),
        "thresholds": config.thresholds.to_dict(),
        "train_seeds": list(config.train_seeds),
        "enable_thinking": config.enable_thinking,
        "git_sha": git_sha,
    }
    if pairs:
        manifest["pairs"] = {
            cid: spec.to_dict() for cid, spec in dict(pairs).items()
        }
    if extra:
        manifest["extra"] = dict(extra)
    manifest["manifest_hash"] = stable_hash(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    return manifest


def write_manifest(manifest: Mapping[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "MANIFEST.json"
    path.write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


__all__ = [
    "C1Budget",
    "ContractError",
    "HarnessPairSpec",
    "PAIR_MODES",
    "PAIR_MODE_ADD",
    "PAIR_MODE_LEGACY",
    "PROBE_OBJECTIVE",
    "PROJECTION_DIRECT_ONLY",
    "ProbeCConfig",
    "SCHEMA_VERSION",
    "SPLIT_AUDIT",
    "SPLIT_NAMES",
    "SPLIT_SELECT",
    "SPLIT_TASK_TEST",
    "SPLIT_TRAIN",
    "STATE_SOURCE_REAL",
    "SplitSpec",
    "Thresholds",
    "UNRESOLVED_REVISION",
    "base_contract_hash",
    "build_manifest",
    "component_revision_of",
    "load_config",
    "make_add_pair",
    "make_legacy_full_minus_pair",
    "write_manifest",
]
