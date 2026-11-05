"""Probe C orchestration: audit → collect → cache → screen → confirm.

CLI (target interface from the implementation guide):

    python -m trim.probes.probe_c_runner audit   --config configs/probes/probe_c_qwen35_9b.json
    python -m trim.probes.probe_c_runner collect --config ...
    python -m trim.probes.probe_c_runner cache   --config ...
    python -m trim.probes.probe_c_runner screen  --config ...
    python -m trim.probes.probe_c_runner confirm --config ...

Design rules enforced here:
- No catch-all fallback: FakeSearchEnv, synthetic Teacher, CE, empty
  supervision and default-Full masks are never substituted on error.
- Infrastructure failures are recorded as ``INFRA_FAILURE`` with bounded
  retry left to the caller; original attempts keep their idempotency keys.
- The audit stage refuses unresolved revisions, unfrozen splits, unsupported
  model loaders, empty targets and wrong mask pairs *before* any expensive
  stage can start.
- GPU/model work is injected through ``RuntimeHooks`` (config ``hooks`` as
  ``module:attr`` specs). A missing hook is a configuration error, not a
  reason to fake a run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trim.eval.probe_c_metrics import (
    MISSING,
    CandidateScore,
    GenerationRecord,
    NllScore,
    aggregate_js,
    aggregate_nll,
    judge_c1,
    regen_stats,
    score_candidates,
    score_teacher_nll,
    sequence_logp_sum,
)
from trim.probes.probe_c_cache import (
    CachedSupervisionRow,
    FrozenTeacherCache,
    FunnelCounts,
    InfraFailure,
    STATUS_READY,
    TeacherTarget,
    build_real_teacher_cache,
    default_teacher_executor,
    write_funnel_jsonl,
)
from trim.probes.probe_c_contract import (
    ContractError,
    HarnessPairSpec,
    ProbeCConfig,
    SPLIT_AUDIT,
    SPLIT_SELECT,
    SPLIT_TRAIN,
    build_manifest,
    load_config,
    make_add_pair,
    make_legacy_full_minus_pair,
    write_manifest,
)
from trim.probes.probe_c_data import (
    ProbeCState,
    StrictLoadError,
    load_strict_real_h0_states,
    split_states,
    write_split_audit,
    write_states_jsonl,
)
from trim.probes.probe_c_train import (
    CandidateResetManager,
    MicroUpdateTrainer,
    OBJECTIVE_FROZEN_PROJECTED_GAP,
    TrainRunResult,
)
from trim.state.snapshot import stable_hash

STAGE_AUDIT = "audit"
STAGE_COLLECT = "collect"
STAGE_CACHE = "cache"
STAGE_SCREEN = "screen"
STAGE_CONFIRM = "confirm"
STAGES = (STAGE_AUDIT, STAGE_COLLECT, STAGE_CACHE, STAGE_SCREEN, STAGE_CONFIRM)

REPORT_JSON = "PROBE_C_REPORT.json"
REPORT_MD = "PROBE_C_REPORT.md"


class RunnerError(ContractError):
    pass


class InfraRecorded(Exception):
    """Raised after an INFRA_FAILURE has been written; CLI exits non-zero."""


# ---------------------------------------------------------------------------
# injectable runtime hooks
# ---------------------------------------------------------------------------


@dataclass
class RuntimeHooks:
    """Everything that needs a model / environment. All optional but explicit.

    - ``collect_states(cfg) -> list[raw rows]``: real H0 rollout collection.
    - ``teacher_logp_fn(cfg, pair) -> TeacherLogpFn``: frozen-theta0 scoring.
    - ``student_logp_fn(backend) -> logp fn``: current-Student scoring.
    - ``student_backend_factory(cfg, seed) -> StudentBackend``: resets by
      recreating from theta0, or use ``CandidateResetManager``.
    - ``make_candidate_set(state, pair, rows) -> list[(id, action)]``: fixed
      K(s) (H0 base + H_m projected + same-tool legal variants).
    - ``regenerate_fn(backend, state, pair) -> GenerationRecord``: fresh
      generation + real parser/schema/visibility checks.
    - ``closed_loop_eval_fn(cfg, backend, query_ids, tag) -> dict``: small
      full H0 episode eval (degradation guard) / C2 task eval.
    - ``c2_runner(cfg, component_id, ...) -> dict``: online 8-step HarnessOPD
      with formal semantics (rollout refresh, real old logp, CISPO).
    """

    collect_states: Callable[[ProbeCConfig], Sequence[Mapping[str, Any]]] | None = None
    teacher_logp_fn: Callable[[ProbeCConfig, HarnessPairSpec], Any] | None = None
    student_backend_factory: Callable[[ProbeCConfig, int], Any] | None = None
    student_logp_fn: Callable[[Any], Any] | None = None
    make_candidate_set: Callable[
        [ProbeCConfig], Callable[..., Sequence[tuple[str, dict[str, Any]]]]
    ] | None = None
    make_teacher_prompt_fn: Callable[[ProbeCConfig, HarnessPairSpec], Any] | None = None
    load_encoder: Callable[[ProbeCConfig], Any] | None = None
    regenerate_fn: Callable[..., GenerationRecord] | None = None
    closed_loop_eval_fn: Callable[..., dict[str, Any]] | None = None
    c2_runner: Callable[..., dict[str, Any]] | None = None
    verify_model_loader: Callable[[ProbeCConfig], dict[str, Any]] | None = None


def _load_dotted(spec: str) -> Any:
    mod_name, _, attr = str(spec).partition(":")
    if not mod_name or not attr:
        raise RunnerError(f"hook spec must be 'module:attr', got {spec!r}")
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        raise RunnerError(f"cannot import hook module {mod_name!r}: {exc}") from exc
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise RunnerError(f"hook {attr!r} not found in {mod_name!r}") from exc


def hooks_from_config(raw: Mapping[str, Any]) -> RuntimeHooks:
    """Load hooks declared in config ``"hooks": {"name": "module:attr"}``."""
    specs = dict(raw.get("hooks") or {})
    kwargs: dict[str, Any] = {}
    for name, spec in specs.items():
        if not hasattr(RuntimeHooks, name):
            raise RunnerError(f"unknown hook name {name!r}")
        kwargs[name] = _load_dotted(str(spec))
    return RuntimeHooks(**kwargs)


# ---------------------------------------------------------------------------
# cost ledger + transactions
# ---------------------------------------------------------------------------


class CostLedger:
    """COST_LEDGER.jsonl: per-stage resource account (§17)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        stage: str,
        component_id: str = "",
        train_seed: int | None = None,
        fixed_budget: int | None = None,
        gpu_hours: float | None = None,
        n_task_calls: int | None = None,
        n_teacher_calls: int | None = None,
        n_prefill_tokens: int | None = None,
        n_processed_tokens: int | None = None,
        peak_mem_gb: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        row = {
            "stage": stage,
            "component_id": component_id,
            "train_seed": train_seed,
            "fixed_budget": fixed_budget,
            "gpu_hours": gpu_hours,
            "n_task_calls": n_task_calls,
            "n_teacher_calls": n_teacher_calls,
            "n_prefill_tokens": n_prefill_tokens,
            "n_processed_tokens": n_processed_tokens,
            "peak_mem_gb": peak_mem_gb,
            **dict(extra or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_transaction(
    out_dir: Path,
    *,
    component_id: str,
    train_seed: int,
    fixed_budget: int,
    payload: Mapping[str, Any],
) -> Path:
    """Idempotent checkpointed write: same (component, seed, budget) overwrites
    only its own transaction file, never mixes with another attempt."""
    tx = out_dir / "transactions" / component_id / f"seed{train_seed}_budget{fixed_budget}.json"
    tx.parent.mkdir(parents=True, exist_ok=True)
    tmp = tx.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(tx)
    return tx


# ---------------------------------------------------------------------------
# fixed-panel evaluation (C1 measurement on held-out states)
# ---------------------------------------------------------------------------


@dataclass
class PanelResult:
    checkpoint: str
    js: dict[str, Any]
    nll: dict[str, Any]
    name_js: dict[str, Any]
    regen: dict[str, Any]
    per_state: list[dict[str, Any]] = field(default_factory=list)
    # Lossless per-generation records + stats object (guide §8.4): judge_c1
    # needs the RegenStats, not only the summary dict.
    regen_stats_obj: Any = None
    generation_records: list[GenerationRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("regen_stats_obj", None)
        payload["generation_records"] = [g.to_dict() for g in self.generation_records]
        return payload

    @property
    def regen_obj(self) -> Any:
        return self.regen_stats_obj if self.regen_stats_obj is not None else regen_stats(self.generation_records)


def evaluate_fixed_panel(
    *,
    checkpoint: str,
    states: Sequence[ProbeCState],
    teacher_targets: Mapping[str, TeacherTarget],
    student_logp_fn: Callable[[Sequence[int], Sequence[int]], Sequence[float]],
    candidate_sets: Mapping[str, Sequence[tuple[str, dict[str, Any]]]],
    z_teacher_by_state: Mapping[str, Mapping[str, float]],
    regenerate_fn: Callable[..., GenerationRecord] | None = None,
    backend: Any = None,
    pair: HarnessPairSpec | None = None,
    regen_limit: int | None = None,
) -> PanelResult:
    """Pre/post evaluation on a fixed panel. K(s) is NOT rebuilt post-training.

    ``regen_limit`` caps fresh *generation* attempts (pilot time bound); scoring
    still covers every state. Capped counts are reported, never silently shrunk.
    """
    js_scores: list[CandidateScore] = []
    nll_scores: list[NllScore] = []
    name_rows: list[tuple[str, float]] = []
    gen_records: list[GenerationRecord] = []
    per_state: list[dict[str, Any]] = []
    n_regen = 0

    for state in states:
        state_js: float | None = None
        state_nll: float | None = None
        state_name_js: float | None = None
        kset = candidate_sets.get(state.state_id) or []
        if len(kset) >= 2:
            z_student = {
                cand_id: sequence_logp_sum(
                    student_logp_fn,
                    prompt_ids=state.effective_student_prompt_ids,
                    target_ids=_action_ids_of(kset, cand_id),
                )
                for cand_id, _action in kset
            }
            z_teacher = dict(z_teacher_by_state.get(state.state_id) or {})
            names = {cand_id: str(a.get("name") or "") for cand_id, a in kset}
            score = score_candidates(
                state_id=state.state_id,
                query_id=state.query_id,
                candidate_ids=[cand_id for cand_id, _ in kset],
                z_teacher=z_teacher,
                z_student=z_student,
                candidate_names=names,
            )
            js_scores.append(score)
            state_js = score.js
            state_name_js = score.name_js
            if score.name_js is not None:
                name_rows.append((state.query_id, score.name_js))
        target = teacher_targets.get(state.state_id)
        if target is not None:
            s_logp = list(
                student_logp_fn(
                    state.effective_student_prompt_ids, target.target_action_ids
                )
            )
            nll = score_teacher_nll(
                state_id=state.state_id,
                query_id=state.query_id,
                token_logprobs=s_logp,
                weights=target.weights,
            )
            nll_scores.append(nll)
            state_nll = nll.nll
        if regenerate_fn is not None and (
            regen_limit is None or n_regen < int(regen_limit)
        ):
            gen_records.append(
                regenerate_fn(backend, state, pair, checkpoint=checkpoint)
            )
            n_regen += 1
        per_state.append(
            {
                "state_id": state.state_id,
                "query_id": state.query_id,
                "checkpoint": checkpoint,
                "js": state_js,
                "nll": state_nll,
                "name_js": state_name_js,
            }
        )

    if name_rows:
        name_per_query: dict[str, list[float]] = {}
        for qid, v in name_rows:
            name_per_query.setdefault(qid, []).append(float(v))
        name_macro = sum(
            sum(vs) / len(vs) for vs in name_per_query.values()
        ) / len(name_per_query)
        name_agg: dict[str, Any] = {
            "d_k": name_macro,
            "d_k_macro": name_macro,
            "d_k_micro": sum(v for _, v in name_rows) / len(name_rows),
            "note": "name-aggregated inside K; candidate-count bias caveat applies",
        }
    else:
        name_agg = {"d_k": MISSING}
    return PanelResult(
        checkpoint=checkpoint,
        js=aggregate_js(js_scores),
        nll=aggregate_nll(nll_scores),
        name_js=name_agg,
        regen=regen_stats(gen_records).to_dict(),
        per_state=per_state,
        regen_stats_obj=regen_stats(gen_records),
        generation_records=gen_records,
    )


def _action_ids_of(
    kset: Sequence[tuple[str, dict[str, Any]]], cid: str
) -> list[int]:
    for k, action in kset:
        if k == cid:
            ids = action.get("token_ids")
            if not isinstance(ids, list) or not ids:
                raise RunnerError(
                    f"candidate {cid!r} has no native token ids; candidates must "
                    "be model-native encoded before scoring"
                )
            return [int(t) for t in ids]
    raise RunnerError(f"candidate {cid!r} missing from K(s)")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def _pairs_for_config(cfg: ProbeCConfig) -> dict[str, HarnessPairSpec]:
    pairs: dict[str, HarnessPairSpec] = {}
    for cid in cfg.components:
        members = list(cfg.members_of(cid))
        if cfg.pair_mode == "add_from_h0":
            pairs[cid] = make_add_pair(
                members,
                harness=cfg.harness,
                component_revision=str(cfg.raw.get("component_revisions", {}).get(cid, ""))
                or "",
                git_sha=str(cfg.raw.get("git_sha") or ""),
                file_hashes=cfg.raw.get("component_file_hashes", {}).get(cid),
            )
        else:
            pairs[cid] = make_legacy_full_minus_pair(
                members,
                harness=cfg.harness,
                git_sha=str(cfg.raw.get("git_sha") or ""),
                file_hashes=cfg.raw.get("component_file_hashes", {}).get(cid),
            )
    return pairs


def stage_audit(cfg: ProbeCConfig, *, out_dir: Path, hooks: RuntimeHooks) -> dict[str, Any]:
    """Refuse bad contracts before any expensive stage."""
    problems: list[str] = []
    # model revision / loader
    try:
        cfg.require_frozen_model(stage=STAGE_AUDIT)
    except ContractError as exc:
        problems.append(str(exc))
    if hooks.verify_model_loader is not None:
        try:
            info = hooks.verify_model_loader(cfg)
            if not info.get("supported", False):
                problems.append(
                    f"unsupported model loader for {cfg.model_id!r}: {info}"
                )
        except Exception as exc:
            problems.append(f"model loader verification failed: {exc}")
    else:
        if not str(cfg.raw.get("model_path") or "").strip():
            problems.append(
                "model_path not configured and no verify_model_loader hook; "
                "cannot verify model loader support"
            )
    # splits
    if not cfg.splits.frozen:
        problems.append("splits are not frozen")
    # mask pairs
    pairs: dict[str, HarnessPairSpec] = {}
    try:
        pairs = _pairs_for_config(cfg)
        for cid, pair in pairs.items():
            want = tuple(sorted(cfg.members_of(cid)))
            if pair.diff_components() != want:
                problems.append(
                    f"mask pair for {cid!r} diffs {pair.diff_components()}, want {want}"
                )
    except ContractError as exc:
        problems.append(f"mask pair construction failed: {exc}")
    # empty targets / unregistered teachers (per coalition member)
    from trim.training.four_cell_runtime import TEACHER_REGISTRY

    member_status: dict[str, list[str]] = {}
    for cid in cfg.components:
        members = list(cfg.members_of(cid))
        member_status[cid] = [
            m for m in members if m not in TEACHER_REGISTRY
        ]
    unregistered = sorted({m for bad in member_status.values() for m in bad})
    if not cfg.components:
        problems.append("empty target component list")

    report = {
        "stage": STAGE_AUDIT,
        "ok": not problems,
        "problems": problems,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "pairs": {c: p.to_dict() for c, p in pairs.items()},
        "unregistered_teachers": unregistered,
        "unregistered_by_candidate": member_status,
        "note": "unregistered components are excluded from screening, not "
        "labelled unlearnable",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    if problems:
        raise RunnerError("audit failed: " + "; ".join(problems))
    manifest = build_manifest(cfg, pairs=pairs)
    write_manifest(manifest, out_dir)
    return report


def stage_collect(
    cfg: ProbeCConfig, *, out_dir: Path, hooks: RuntimeHooks
) -> dict[str, Any]:
    """Strict-validate real H0 states and freeze the split audit."""
    if hooks.collect_states is None:
        raise RunnerError(
            "collect hook not configured; set hooks.collect_states to a real "
            "H0 rollout collector (no FakeSearchEnv fallback)"
        )
    raw_rows = list(hooks.collect_states(cfg))
    states = load_strict_real_h0_states(raw_rows)
    by_split = split_states(states)
    # Cap per split by *reporting actual counts*; never duplicate rows.
    report: dict[str, Any] = {
        "stage": STAGE_COLLECT,
        "n_states": len(states),
        "per_split": {
            name: {
                "n_states": len(rows),
                "n_queries": len({s.query_id for s in rows}),
                "target_max_states": cfg.splits.max_states().get(name),
            }
            for name, rows in by_split.items()
        },
    }
    write_states_jsonl(states, out_dir / "states" / "all_states.jsonl")
    for name, rows in by_split.items():
        if rows:
            write_states_jsonl(rows, out_dir / "states" / f"{name}.jsonl")
    write_split_audit(states, out_dir)
    (out_dir / "COLLECT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return report


def _load_states_for_stage(cfg: ProbeCConfig, out_dir: Path) -> list[ProbeCState]:
    path = out_dir / "states" / "all_states.jsonl"
    if not path.is_file():
        raise RunnerError(f"states not found: {path}; run collect first")
    return load_strict_real_h0_states(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    )


def stage_cache(
    cfg: ProbeCConfig, *, out_dir: Path, hooks: RuntimeHooks
) -> dict[str, Any]:
    """C0 funnel + frozen Teacher cache per component (shared states)."""
    if hooks.teacher_logp_fn is None:
        raise RunnerError(
            "teacher_logp_fn hook not configured; frozen-theta0 Teacher scoring "
            "is required (no current-Student substitution)"
        )
    states = _load_states_for_stage(cfg, out_dir)
    by_split = split_states(states)
    # C0 pilot subset: up to 16 queries per split with ALL their decision points
    # (early / mid / pre-stop). Turn-0-only sampling hides every mid-episode
    # trigger (verify_tool / budget / rerank / compress) and yields fake
    # NO_TRIGGER.
    pilot_n = int(cfg.raw.get("c0_pilot_queries", 16))
    pilot: list[ProbeCState] = []
    for name in (SPLIT_TRAIN, SPLIT_SELECT):
        rows = by_split.get(name) or []
        by_q: dict[str, list[ProbeCState]] = {}
        for st in rows:
            by_q.setdefault(st.query_id, []).append(st)
        for i, qid in enumerate(sorted(by_q)):
            if i >= pilot_n:
                break
            pilot.extend(by_q[qid])
    component_status: dict[str, Any] = {}
    for cid in cfg.components:
        pair = _pairs_for_config(cfg)[cid]
        members = list(cfg.members_of(cid))
        cache = FrozenTeacherCache()
        logp_fn = hooks.teacher_logp_fn(cfg, pair)
        teacher_prompt_fn = (
            hooks.make_teacher_prompt_fn(cfg, pair)
            if hooks.make_teacher_prompt_fn is not None
            else None
        )
        try:
            rows, funnel = build_real_teacher_cache(
                pilot,
                pair,
                component_id=cid,
                executor=default_teacher_executor(",".join(members)),
                encoder=_encoder_from_hooks(cfg, hooks, pair),
                teacher_logp_fn=logp_fn,
                cache=cache,
                teacher_prompt_fn=teacher_prompt_fn,
            )
        except InfraFailure as exc:
            _write_infra(out_dir, cid, STAGE_CACHE, exc)
            component_status[cid] = {"status": "INFRA_FAILURE", "error": str(exc)}
            continue
        status = funnel.status(
            min_covered_queries=cfg.thresholds.min_covered_queries,
            min_supervised_tokens=cfg.thresholds.min_supervised_tokens,
        )
        write_funnel_jsonl(funnel, out_dir / "cache" / cid)
        cache.write_manifest(out_dir / "cache" / cid)
        component_status[cid] = {
            "status": status,
            "funnel": funnel.to_dict(),
            "n_cached_targets": len(rows),
            "pair_hash": pair.resolved_pair_hash(),
        }
        # Persist cached targets for screen (without full vocab logits).
        _write_cache_rows(out_dir / "cache" / cid, rows)
    report = {"stage": STAGE_CACHE, "components": component_status}
    (out_dir / "CACHE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return report


def _encoder_from_hooks(
    cfg: ProbeCConfig, hooks: RuntimeHooks, pair: HarnessPairSpec
) -> Callable[[ProbeCState, Mapping[str, Any]], tuple[list[int], list[bool]]]:
    """Native action encoding. A generic `to=tool` text substitute is forbidden."""
    if hooks.load_encoder is None:
        raise RunnerError(
            "no encoding hook configured; native model encoding is required"
        )

    def _encode(
        state: ProbeCState, target_action: Mapping[str, Any]
    ) -> tuple[list[int], list[bool]]:
        from trim.training.action_encoding import encode_supervised_action

        if hooks.load_encoder is None:
            raise RunnerError(
                "load_encoder hook not configured; native model encoding is required"
            )
        enc = hooks.load_encoder(cfg)
        sup = encode_supervised_action(
            target_action=dict(target_action),
            target_text="",
            encode=lambda text: enc.encode(text) if hasattr(enc, "encode") else [],
            model_enc=enc,
        )
        return list(sup.target_ids), list(sup.supervision_mask)

    return _encode


def _write_cache_rows(out_dir: Path, rows: Sequence[CachedSupervisionRow]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "CACHED_TARGETS.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "state_id": row.state.state_id,
                "query_id": row.state.query_id,
                "target": {
                    k: v
                    for k, v in row.target.__dict__.items()
                },
                "funnel": row.funnel.to_dict(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _load_cache_rows(out_dir: Path, cid: str) -> list[dict[str, Any]]:
    path = out_dir / "cache" / cid / "CACHED_TARGETS.jsonl"
    if not path.is_file():
        raise RunnerError(f"cache rows not found for {cid}: {path}; run cache first")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_infra(out_dir: Path, cid: str, stage: str, exc: Exception) -> None:
    d = out_dir / "infra_failures"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cid}_{stage}.json").write_text(
        json.dumps(
            {
                "status": "INFRA_FAILURE",
                "component_id": cid,
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "retry_policy": "bounded retry only for network/resource errors; "
                "original attempt preserved",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def stage_screen(
    cfg: ProbeCConfig, *, out_dir: Path, hooks: RuntimeHooks
) -> dict[str, Any]:
    """C1: fixed-Teacher micro-updates 4→8 (+16 borderline) + fixed panel eval."""
    if hooks.student_backend_factory is None or hooks.student_logp_fn is None:
        raise RunnerError(
            "student_backend_factory / student_logp_fn hooks are required for screen"
        )
    states = _load_states_for_stage(cfg, out_dir)
    by_split = split_states(states)
    select_states = by_split.get(SPLIT_SELECT) or []
    if not select_states:
        raise RunnerError("no C_select states; run collect first")

    budget = cfg.budget_spec()
    rows_out: list[dict[str, Any]] = []
    component_status: dict[str, Any] = {}

    # Per-candidate panel data (frozen before any training).
    panels: dict[str, dict[str, Any]] = {}
    for cid in cfg.components:
        pair = _pairs_for_config(cfg)[cid]
        cache_payload = _load_cache_rows(out_dir, cid)
        funnel_path = out_dir / "cache" / cid / "TEACHER_FUNNEL.jsonl"
        support_status = STATUS_READY
        if funnel_path.is_file():
            fr = [json.loads(x) for x in funnel_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            funnel = FunnelCounts(component_id=cid, rows=[])  # rebuilt below
            from trim.probes.probe_c_cache import FunnelRow

            funnel.rows = [FunnelRow(**{k: v for k, v in r.items() if k in FunnelRow.__dataclass_fields__}) for r in fr]
            support_status = funnel.status(
                min_covered_queries=cfg.thresholds.min_covered_queries,
                min_supervised_tokens=cfg.thresholds.min_supervised_tokens,
            )
        if support_status != STATUS_READY:
            component_status[cid] = {
                "status": support_status,
                "reason": "support funnel not READY; skipped C1",
            }
            continue

        targets = {
            p["target"]["state_id"]: TeacherTarget(**p["target"]) for p in cache_payload
        }
        # Split targets by the split of their state (guide §7.1): C_train rows
        # feed gradients; C_select same-state targets build the held-out K(s).
        # The legacy code passed train_rows to the candidate constructor for
        # held-out states, whose state_id filter then dropped everything —
        # held-out K(s) silently lost its Teacher target.
        targets_by_state_by_split: dict[str, dict[str, list[TeacherTarget]]] = {
            SPLIT_TRAIN: {},
            SPLIT_SELECT: {},
        }
        for sid, tgt in targets.items():
            st = _find_state(states, sid)
            targets_by_state_by_split.setdefault(st.split, {}).setdefault(sid, []).append(tgt)
        select_targets_by_state = targets_by_state_by_split.get(SPLIT_SELECT, {})
        # C1 trains only on C_train rows; the C_select panel is the query-disjoint
        # eval side (its cached targets feed Teacher-sample NLL and K(s)).
        train_rows = [
            CachedSupervisionRow(
                state=_find_state(states, sid),
                target=targets[sid],
                funnel=None,
            )
            for sid in sorted(targets_by_state_by_split.get(SPLIT_TRAIN, {}))
        ]
        train_qids = {r.state.query_id for r in train_rows}
        heldout = [s for s in select_states if s.query_id not in train_qids]
        if not heldout:
            raise RunnerError(
                "C_select panel is fully query-overlapped with C_train targets; "
                "split assignment is broken (no held-out states). Refusing the "
                "legacy select_states fallback that silently ran on non-held-out data"
            )
        # Pilot time bounds (reported, never silent): scoring panel + regen cap.
        screen_max_states = int(cfg.raw.get("screen_max_states", 128))
        if len(heldout) > screen_max_states:
            heldout = heldout[:screen_max_states]
        regen_limit = int(cfg.raw.get("regen_max_states", 64))

        # Candidate sets and Teacher candidate scores are frozen before training.
        if hooks.make_candidate_set is None:
            raise RunnerError("make_candidate_set hook is required for screen")
        candidate_fn = hooks.make_candidate_set(cfg)
        teacher_logp = hooks.teacher_logp_fn(cfg, pair)
        teacher_prompt_fn = (
            hooks.make_teacher_prompt_fn(cfg, pair)
            if hooks.make_teacher_prompt_fn is not None
            else None
        )
        candidate_sets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        z_teacher_by_state: dict[str, dict[str, float]] = {}
        target_in_K: dict[str, bool] = {}
        for state in heldout:
            # Same-state C_select targets only — never train_rows (§7.1).
            same_state_rows = [
                CachedSupervisionRow(state=state, target=tgt, funnel=None)
                for tgt in select_targets_by_state.get(state.state_id, [])
            ]
            kset = list(candidate_fn(state, pair, same_state_rows))
            candidate_sets[state.state_id] = kset
            kset_ids = {cand_id for cand_id, _ in kset}
            _tgt = select_targets_by_state.get(state.state_id, [])
            target_in_K[state.state_id] = (
                any(
                    any(
                        cand_action.get("name") == t.target_action.get("name")
                        and cand_action.get("arguments") == t.target_action.get("arguments")
                        for _cand_id, cand_action in kset
                    )
                    for t in _tgt
                )
                if _tgt
                else False
            )
            t_prompt = (
                list(teacher_prompt_fn(state, None))
                if teacher_prompt_fn is not None
                else list(state.effective_student_prompt_ids)
            )
            z_teacher_by_state[state.state_id] = {
                cand_id: float(
                    sum(
                        float(x)
                        for x in teacher_logp(t_prompt, _action_ids_of(kset, cand_id))
                    )
                )
                for cand_id, _ in kset
            }
        panels[cid] = {
            "pair": pair,
            "support_status": support_status,
            "targets": targets,
            "train_rows": train_rows,
            "heldout": heldout,
            "candidate_sets": candidate_sets,
            "z_teacher_by_state": z_teacher_by_state,
            "regen_limit": regen_limit,
            "target_in_K": target_in_K,
            "n_select_targets": len(select_targets_by_state),
        }

    # Seed-outer loop: one resident Student per seed, exact θ0 restart per candidate.
    for seed in cfg.train_seeds:
        backend = hooks.student_backend_factory(cfg, int(seed))
        reset = CandidateResetManager(backend)
        theta0 = reset.capture_theta0()
        trainer = MicroUpdateTrainer(
            backend,
            gate_beta=budget.gate_beta,
            loss_scale=budget.loss_scale,
            action_tokens_per_optimizer_step=budget.action_tokens_per_optimizer_step,
        )
        for cid in cfg.components:
            panel = panels.get(cid)
            if panel is None:
                continue
            pair = panel["pair"]
            reset.reset_exact(int(seed))
            reset.verify_theta0()
            pre = evaluate_fixed_panel(
                checkpoint="pre",
                states=panel["heldout"],
                teacher_targets=panel["targets"],
                student_logp_fn=hooks.student_logp_fn(backend),
                candidate_sets=panel["candidate_sets"],
                z_teacher_by_state=panel["z_teacher_by_state"],
                regenerate_fn=hooks.regenerate_fn,
                backend=backend,
                pair=pair,
                regen_limit=panel["regen_limit"],
            )
            seed_rows: list[dict[str, Any]] = []
            step_offset = 0
            for stage_budget in sorted(set(int(c) for c in budget.checkpoints if c > 0)):
                if stage_budget > budget.borderline_max_step:
                    break
                run: TrainRunResult = trainer.train_to_budget(
                    panel["train_rows"],
                    target_steps=stage_budget - step_offset,
                    component_id=cid,
                    train_seed=int(seed),
                    checkpoint_steps=[stage_budget],
                    objective=OBJECTIVE_FROZEN_PROJECTED_GAP,
                    start_step=step_offset,
                    theta0_checksum=theta0,
                )
                step_offset = stage_budget
                reset.verify_updated()
                post = evaluate_fixed_panel(
                    checkpoint=f"post@{stage_budget}",
                    states=panel["heldout"],
                    teacher_targets=panel["targets"],
                    student_logp_fn=hooks.student_logp_fn(backend),
                    candidate_sets=panel["candidate_sets"],
                    z_teacher_by_state=panel["z_teacher_by_state"],
                    regenerate_fn=hooks.regenerate_fn,
                    backend=backend,
                    pair=pair,
                    regen_limit=panel["regen_limit"],
                )
                verdict = judge_c1(
                    pre_js=pre.js.get("d_k", MISSING),
                    post_js=post.js.get("d_k", MISSING),
                    pre_nll=pre.nll.get("nll_macro", MISSING),
                    post_nll=post.nll.get("nll_macro", MISSING),
                    pre_name_js=pre.name_js.get("d_k", MISSING),
                    post_name_js=post.name_js.get("d_k", MISSING),
                    pre_regen=pre.regen_obj,
                    post_regen=post.regen_obj,
                    support_status=panel["support_status"],
                    delta_nll=cfg.thresholds.delta_nll,
                    delta_dk=cfg.thresholds.delta_dk,
                    max_dk_regression=float(
                        cfg.raw.get("max_dk_regression",
                        cfg.thresholds.delta_dk)
                    ),
                    eps_name_js=cfg.thresholds.eps_name_js,
                    eps_invalid=cfg.thresholds.eps_invalid,
                    require_generation_guard=bool(
                        cfg.raw.get("require_generation_guard", True)
                    ),
                    absolute_quality_status=str(
                        cfg.raw.get("absolute_quality_status", "QUALITY_UNRESOLVED")
                    ),
                )
                row = {
                    "model_id": cfg.model_id,
                    "component_id": cid,
                    "train_seed": int(seed),
                    "fixed_budget": stage_budget,
                    "status": verdict.status,
                    "pair_hash": pair.resolved_pair_hash(),
                    "covered_queries": len({s.query_id for s in panel["heldout"]}),
                    "supervised_tokens": run.total_tokens,
                    "d_k_pre": pre.js.get("d_k", MISSING),
                    "d_k_post": post.js.get("d_k", MISSING),
                    "nll_pre": pre.nll.get("nll_macro", MISSING),
                    "nll_post": post.nll.get("nll_macro", MISSING),
                    "name_js": post.name_js.get("d_k", MISSING),
                    "generated_invalid_post": post.regen.get("invalid_generated_rate", MISSING),
                    "generated_invalid_pre": pre.regen.get("invalid_generated_rate", MISSING),
                    "guard_status": verdict.detail.get("guard_status", MISSING),
                    "target_in_K_rate": (
                        sum(1 for v in panel["target_in_K"].values() if v)
                        / max(1, len(panel["target_in_K"]))
                    ),
                    "g": MISSING,
                    "e": MISSING,
                    "gpu_hours": MISSING,
                    "stage": "C1",
                    "ci": MISSING,
                    "verdict_detail": verdict.to_dict(),
                }
                seed_rows.append(row)
                rows_out.append(row)
                save_transaction(
                    out_dir,
                    component_id=cid,
                    train_seed=int(seed),
                    fixed_budget=stage_budget,
                    payload={"pre": pre.to_dict(), "post": post.to_dict(), "row": row},
                )
                # Per-generation raw records land on disk before any verdict is
                # consumed (guide §15: per-state/per-generation first, summary later).
                gen_dir = out_dir / "generations" / cid / f"seed{seed}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                with (gen_dir / f"budget{stage_budget}_pre.jsonl").open(
                    "w", encoding="utf-8"
                ) as fh:
                    for g in pre.generation_records:
                        fh.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                with (gen_dir / f"budget{stage_budget}_post.jsonl").open(
                    "w", encoding="utf-8"
                ) as fh:
                    for g in post.generation_records:
                        fh.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                # Contract: all READY candidates run 4→8 on one optimizer
                # trajectory; only pre-registered BORDERLINE candidates continue
                # to borderline_max_step (16).
                if stage_budget >= budget.borderline_max_step:
                    break
                if stage_budget >= 8 and verdict.status != "BORDERLINE":
                    break
            component_status.setdefault(cid, []).append(
                {"train_seed": int(seed), "rows": seed_rows, "theta0": theta0}
            )

    report = {
        "stage": STAGE_SCREEN,
        "rows": rows_out,
        "components": component_status,
        "row_schema": "model × component × train_seed × fixed_budget",
    }
    (out_dir / "SCREEN.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    _write_report(out_dir, rows_out, extra={"stage": STAGE_SCREEN})
    return report


def _find_state(states: Sequence[ProbeCState], state_id: str) -> ProbeCState:
    for st in states:
        if st.state_id == state_id:
            return st
    raise RunnerError(f"state {state_id!r} not found in loaded states")


def stage_confirm(
    cfg: ProbeCConfig, *, out_dir: Path, hooks: RuntimeHooks
) -> dict[str, Any]:
    """C2: 8-step online confirmation from θ0 with formal HarnessOPD semantics.

    Control groups (§10): No-update / Runtime-m / RL-only / HarnessOPD-m /
    Same-view control (representative components only).
    """
    if hooks.c2_runner is None:
        raise RunnerError(
            "c2_runner hook not configured; C2 must use the existing formal "
            "HarnessOPD trainer semantics (no offline replay labelled as online RL)"
        )
    screen_path = out_dir / "SCREEN.json"
    if not screen_path.is_file():
        raise RunnerError("SCREEN.json missing; run screen before confirm")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    # Fixed-budget summary (guide §9.4): C2 picks come from the pre-registered
    # primary budget, aggregated across seeds — never from "any row once
    # PROMOTEd" (seed11@4 must not override the 8-step result).
    primary_budget = int(cfg.raw.get("primary_budget", 8))
    by_component: dict[str, list[dict[str, Any]]] = {}
    for r in screen.get("rows", []):
        if int(r.get("fixed_budget") or 0) == primary_budget:
            by_component.setdefault(str(r["component_id"]), []).append(r)

    def _summary_status(rows: list[dict[str, Any]]) -> str:
        statuses = [str(r.get("status")) for r in rows]
        if statuses and all(s == "PROMOTE" for s in statuses):
            return "PROMOTE"
        if any(s in {"GUARD_FAIL", "DEPRIORITIZE"} for s in statuses) and not any(
            s == "PROMOTE" for s in statuses
        ):
            return "GUARD_OR_DEPRIORITIZED"
        return "BORDERLINE"

    promote = sorted(
        cid for cid, rows in by_component.items() if _summary_status(rows) == "PROMOTE"
    )
    borderline = sorted(
        cid for cid, rows in by_component.items() if _summary_status(rows) == "BORDERLINE"
    )
    picks = promote[:2]
    audit_pick = borderline[:1]
    results: dict[str, Any] = {}
    for cid in list(dict.fromkeys(picks + audit_pick)):
        try:
            results[cid] = hooks.c2_runner(
                cfg,
                component_id=cid,
                from_theta0=True,
                n_updates=8,
                controls=[
                    "no_update",
                    "runtime_m",
                    "rl_only",
                    "harnessopd_m",
                    "same_view_control",
                ],
            )
        except InfraFailure as exc:
            _write_infra(out_dir, cid, STAGE_CONFIRM, exc)
            results[cid] = {"status": "INFRA_FAILURE", "error": str(exc)}
    report = {
        "stage": STAGE_CONFIRM,
        "promoted": promote,
        "borderline_audit": audit_pick,
        "results": results,
        "note": "C2 restarts from theta0; C1 weights are not carried over",
    }
    (out_dir / "CONFIRM.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _write_report(out_dir: Path, rows: Sequence[Mapping[str, Any]], *, extra: Mapping[str, Any]) -> None:
    payload = {
        "rows": list(rows),
        **dict(extra),
        "report_hash": stable_hash({"rows": list(rows), **dict(extra)}),
    }
    (out_dir / REPORT_JSON).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Probe C report",
        "",
        "Row schema: model × component × train_seed × fixed_budget.",
        "",
        "| component | seed | budget | status | D_K pre→post | NLL pre→post | invalid post | covered q | tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('component_id')} | {r.get('train_seed')} | {r.get('fixed_budget')} "
            f"| {r.get('status')} | {r.get('d_k_pre')}→{r.get('d_k_post')} "
            f"| {r.get('nll_pre')}→{r.get('nll_post')} | {r.get('generated_invalid_post')} "
            f"| {r.get('covered_queries')} | {r.get('supervised_tokens')} |"
        )
    (out_dir / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe_c_runner")
    parser.add_argument(
        "stage",
        choices=list(STAGES),
        help="audit | collect | cache | screen | confirm",
    )
    parser.add_argument("--config", required=True, help="probe_c_*.json contract")
    parser.add_argument("--out", default="", help="output root (default from config)")
    parser.add_argument(
        "--hooks-module",
        default="",
        help="module providing build_hooks() -> RuntimeHooks (overrides config hooks)",
    )
    return parser


def _resolve_hooks(args: argparse.Namespace, cfg: ProbeCConfig) -> RuntimeHooks:
    if args.hooks_module:
        mod = importlib.import_module(args.hooks_module)
        return mod.build_hooks()
    return hooks_from_config(cfg.raw)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, stage=args.stage)
    except ContractError as exc:
        print(f"[probe_c] REFUSED: {exc}")
        return 2
    out_dir = Path(args.out or cfg.output_root or "outputs/probe_c") / cfg.model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        hooks = _resolve_hooks(args, cfg)
    except RunnerError as exc:
        print(f"[probe_c] REFUSED: {exc}")
        return 2
    ledger = CostLedger(out_dir / "COST_LEDGER.jsonl")
    start = time.time()
    try:
        if args.stage == STAGE_AUDIT:
            result = stage_audit(cfg, out_dir=out_dir, hooks=hooks)
        elif args.stage == STAGE_COLLECT:
            result = stage_collect(cfg, out_dir=out_dir, hooks=hooks)
        elif args.stage == STAGE_CACHE:
            result = stage_cache(cfg, out_dir=out_dir, hooks=hooks)
        elif args.stage == STAGE_SCREEN:
            result = stage_screen(cfg, out_dir=out_dir, hooks=hooks)
        else:
            result = stage_confirm(cfg, out_dir=out_dir, hooks=hooks)
    except (ContractError, RunnerError, StrictLoadError) as exc:
        print(f"[probe_c] REFUSED: {exc}")
        return 2
    except InfraFailure as exc:
        _write_infra(out_dir, "ALL", args.stage, exc)
        print(f"[probe_c] INFRA_FAILURE: {exc}")
        return 3
    ledger.record(stage=args.stage, extra={"seconds": time.time() - start})
    print(f"[probe_c] {args.stage} ok -> {out_dir}")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
