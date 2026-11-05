"""Probe C R2: repaired C1, two seeds, 4→8 (guide §16 R2).

Scope: single members with support READY (auto_populate_first_search first;
others enter only when their R1 per-split support passes the threshold).

Hard requirements implemented here (guide R2 list):
- Same objective / LoRA range / LR as the legacy runs (comparable first pass).
- REAL checkpoints at pre/4/8: weights + optimizer + RNG, atomic tmp→verify→
  rename (``CandidateResetManager.save_checkpoint_atomic``).
- One persistent sampler session per (component, seed): 4→8 continues the
  same data stream (``MicroUpdateTrainer.train_until``; 4+4==8 contract).
- Panel metrics recomputed pre/4/8 on the SAME frozen select panel; every
  generation record lands on disk.
- Fixed-budget summary at 8 across both seeds decides the C2 pick; per-seed
  rows are reported alongside (descriptive only at 2 seeds).
- BORDERLINE at 8 with measurable supervision + generation quality may
  extend to 16 (pre-registered borderline_max_step).

Outputs (under OUT/<model_id>/r2/<member>/seed<N>/):
- TRAIN_STEPS.jsonl        per-step records (loss/z_gap/grad_norm/tokens)
- checkpoint_step0000/0004/0008.pt
- PANEL_PRE/POST4/POST8.json
- GENERATIONS_{pre,post4,post8}.jsonl
Plus R2_C1_REPORT.json + R2_C1_SUMMARY.json at the model root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from trim.eval.probe_c_metrics import (
    MISSING,
    RegenStats,
    aggregate_js,
    aggregate_nll,
    judge_c1,
    regen_stats,
)
from trim.probes.probe_c_cache import (
    CachedSupervisionRow,
    STATUS_READY,
    TeacherTarget,
)
from trim.probes.probe_c_contract import (
    SPLIT_SELECT,
    SPLIT_TRAIN,
    load_config,
)
from trim.probes.probe_c_data import (
    ProbeCState,
    load_strict_real_h0_states,
    split_states,
)
from trim.probes.probe_c_r1_smoke import _pair_for_member
from trim.probes.probe_c_runner import (
    PanelResult,
    _action_ids_of,
    evaluate_fixed_panel,
    hooks_from_config,
)
from trim.probes.probe_c_train import (
    CandidateResetManager,
    MicroUpdateTrainer,
    OBJECTIVE_FROZEN_PROJECTED_GAP,
)
from trim.state.snapshot import stable_hash

R2_SCHEMA = "probe_c_r2_c1_v1"


def load_panel_states(cfg: Any, states_dir: Path) -> list[ProbeCState]:
    states_path = states_dir / "states" / "all_states.jsonl"
    if not states_path.is_file():
        states_path = states_dir / "all_states.jsonl"
    if not states_path.is_file():
        raise FileNotFoundError(f"states not found under {states_dir}")
    return load_strict_real_h0_states(
        [json.loads(l) for l in states_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    )


def load_cache_rows(cache_dir: Path) -> list[dict[str, Any]]:
    path = cache_dir / "CACHED_TARGETS.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"cache rows not found: {path}; run R1 first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_member_seed(
    *,
    cfg: Any,
    hooks: Any,
    member: str,
    seed: int,
    panel_train: list[ProbeCState],
    panel_select: list[ProbeCState],
    cache_rows: list[dict[str, Any]],
    out_dir: Path,
    checkpoints: Sequence[int],
    borderline_max_step: int,
    extend_to_16: bool,
    regen_limit: int,
) -> dict[str, Any]:
    pair = _pair_for_member(cfg, member)
    states_by_id = {s.state_id: s for s in panel_train + panel_select}
    targets: dict[str, TeacherTarget] = {}
    train_rows: list[CachedSupervisionRow] = []
    select_targets: dict[str, list[TeacherTarget]] = {}
    for p in cache_rows:
        sid = p["state_id"]
        st = states_by_id.get(sid)
        if st is None:
            continue
        tgt = TeacherTarget(**p["target"])
        targets[sid] = tgt
        if st.split == SPLIT_TRAIN:
            train_rows.append(CachedSupervisionRow(state=st, target=tgt, funnel=None))
        else:
            select_targets.setdefault(sid, []).append(tgt)
    if not train_rows:
        raise RuntimeError(f"{member} seed{seed}: no C_train supervision rows")
    train_qids = {r.state.query_id for r in train_rows}
    heldout = [s for s in panel_select if s.query_id not in train_qids]
    if not heldout:
        raise RuntimeError("no held-out select states (cross-split leakage?)")

    if hooks.student_backend_factory is None or hooks.student_logp_fn is None:
        raise RuntimeError("student_backend_factory + student_logp_fn hooks required")
    backend = hooks.student_backend_factory(cfg, int(seed))
    reset = CandidateResetManager(backend)
    theta0 = reset.capture_theta0()

    ck_dir = out_dir / "checkpoints"
    if hooks.make_candidate_set is None:
        raise RuntimeError("make_candidate_set hook required")
    candidate_fn = hooks.make_candidate_set(cfg)
    teacher_logp = hooks.teacher_logp_fn(cfg, pair)
    # Freeze K(s) + teacher candidate scores before any training (same as runner).
    candidate_sets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    z_teacher_by_state: dict[str, dict[str, float]] = {}
    for state in heldout:
        same_state_rows = [
            CachedSupervisionRow(state=state, target=t, funnel=None)
            for t in select_targets.get(state.state_id, [])
        ]
        kset = list(candidate_fn(state, pair, same_state_rows))
        candidate_sets[state.state_id] = kset
        z_teacher_by_state[state.state_id] = {
            cand_id: float(sum(float(x) for x in teacher_logp(
                list(state.effective_student_prompt_ids), _action_ids_of(kset, cand_id))))
            for cand_id, _ in kset
        }

    budget = cfg.budget_spec()
    trainer = MicroUpdateTrainer(
        backend,
        gate_beta=budget.gate_beta,
        loss_scale=budget.loss_scale,
        action_tokens_per_optimizer_step=budget.action_tokens_per_optimizer_step,
    )

    def _panel(tag: str) -> tuple[PanelResult, Path]:
        res = evaluate_fixed_panel(
            checkpoint=tag,
            states=heldout,
            teacher_targets=targets,
            student_logp_fn=hooks.student_logp_fn(backend),
            candidate_sets=candidate_sets,
            z_teacher_by_state=z_teacher_by_state,
            regenerate_fn=hooks.regenerate_fn,
            backend=backend,
            pair=pair,
            regen_limit=regen_limit,
        )
        gen_path = out_dir / f"GENERATIONS_{tag}.jsonl"
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with gen_path.open("w", encoding="utf-8") as fh:
            for g in res.generation_records:
                fh.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return res, gen_path

    pre, _ = _panel("pre")
    ck0 = reset.save_checkpoint_atomic(ck_dir, tag="step0000_pre")

    session = trainer.open_session(train_rows, train_seed=int(seed))
    budgets = sorted({int(c) for c in checkpoints if c > 0})
    if extend_to_16 and borderline_max_step not in budgets:
        budgets.append(borderline_max_step)
    stage_panels: dict[str, PanelResult] = {}
    stage_results: dict[int, Any] = {}
    all_steps: list[dict[str, Any]] = []
    last_verdict: Any = None
    for abs_step in budgets:
        run = trainer.train_until(
            session,
            train_rows,
            target_absolute_step=abs_step,
            component_id=member,
            train_seed=int(seed),
            checkpoint_steps=[abs_step],
            checkpoint_dir=ck_dir,
        )
        for s in run.steps:
            all_steps.append(s.to_dict())
        post, _ = _panel(f"post{abs_step}")
        stage_panels[f"post{abs_step}"] = post
        verdict = judge_c1(
            pre_js=pre.js.get("d_k", MISSING),
            post_js=post.js.get("d_k", MISSING),
            pre_nll=pre.nll.get("nll_macro", MISSING),
            post_nll=post.nll.get("nll_macro", MISSING),
            pre_name_js=pre.name_js.get("d_k", MISSING),
            post_name_js=post.name_js.get("d_k", MISSING),
            pre_regen=pre.regen_obj,
            post_regen=post.regen_obj,
            support_status=STATUS_READY,
            delta_nll=cfg.thresholds.delta_nll,
            delta_dk=cfg.thresholds.delta_dk,
            max_dk_regression=float(cfg.raw.get("max_dk_regression", cfg.thresholds.delta_dk)),
            eps_name_js=cfg.thresholds.eps_name_js,
            eps_invalid=cfg.thresholds.eps_invalid,
            require_generation_guard=True,
            absolute_quality_status="",
        )
        stage_results[abs_step] = {
            "verdict": verdict.to_dict(),
            "panel": {
                "d_k_pre": pre.js.get("d_k", MISSING),
                "d_k_post": post.js.get("d_k", MISSING),
                "nll_pre": pre.nll.get("nll_macro", MISSING),
                "nll_post": post.nll.get("nll_macro", MISSING),
                "regen_pre": pre.regen,
                "regen_post": post.regen,
                "n_states_scored_nll": post.nll.get("n_states_scored"),
                "n_states_scored_js": post.js.get("n_states_scored"),
            },
            "steps_this_stage": [s.step for s in run.steps],
            "segment_tokens": run.total_tokens,
            "cumulative_tokens": session.consumed_action_tokens,
        }
        last_verdict = verdict
        (out_dir / f"PANEL_POST{abs_step}.json").write_text(
            json.dumps(post.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        # Pre-registered 16-step extension only for BORDERLINE at 8.
        if abs_step >= 8 and verdict.status != "BORDERLINE":
            break

    with (out_dir / "PANEL_PRE.json").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(pre.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    with (out_dir / "TRAIN_STEPS.jsonl").open("w", encoding="utf-8") as fh:
        for s in all_steps:
            fh.write(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "member": member,
        "train_seed": int(seed),
        "theta0_checksum": theta0,
        "checkpoint_pre": ck0["model_weights_checksum"],
        "stages": stage_results,
        "final_status": last_verdict.status if last_verdict else MISSING,
        "n_train_rows": len(train_rows),
        "n_heldout_states": len(heldout),
        "cumulative_tokens": session.consumed_action_tokens,
        "contract_hash": session.contract_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_c_r2_c1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="R2 output root (NEW run)")
    parser.add_argument("--r1-dir", required=True, help="R1 run dir with cache/<member>/")
    parser.add_argument("--states-dir", default="", help="old run with states/ (default cfg.state_source_dir)")
    parser.add_argument("--members", default="auto_populate_first_search")
    parser.add_argument("--seeds", default="11,23")
    parser.add_argument("--min-select-supported-states", type=int, default=8,
                        help="member needs ≥N supervised select states to enter C1")
    parser.add_argument("--extend-16", action="store_true",
                        help="allow pre-registered 16-step extension for BORDERLINE@8")
    parser.add_argument("--regen-limit", type=int, default=16)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, stage="screen")
    model_dir = Path(args.out) / cfg.model_id.replace("/", "_")
    r2_root = model_dir / "r2"
    r2_root.mkdir(parents=True, exist_ok=True)
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    states_dir = Path(args.states_dir or str(cfg.raw.get("state_source_dir") or ""))
    states = load_panel_states(cfg, states_dir)
    by_split = split_states(states)
    # R1 panel replication: same stratified selection, same sizes.
    from trim.probes.probe_c_r1_smoke import select_panel

    panel = select_panel(
        states,
        n_train_queries=int(cfg.raw.get("r2_train_queries", 16)),
        n_select_queries=int(cfg.raw.get("r2_select_queries", 16)),
    )
    panel_train = panel[SPLIT_TRAIN]
    panel_select = panel[SPLIT_SELECT]

    r1_dir = Path(args.r1_dir)
    report: dict[str, Any] = {
        "schema": R2_SCHEMA,
        "members": members,
        "seeds": seeds,
        "panel": {k: {"n_queries": len({s.query_id for s in v}), "n_states": len(v)}
                  for k, v in panel.items()},
        "started_at": time.strftime("%F %T"),
        "members_run": {},
    }
    hooks = hooks_from_config(cfg.raw)
    for member in members:
        cache_rows = load_cache_rows(r1_dir / "cache" / member)
        n_select_sup = sum(
            1 for p in cache_rows
            if states_by_id_split(p, states) == SPLIT_SELECT
        )
        if n_select_sup < int(args.min_select_supported_states):
            report["members_run"][member] = {
                "skipped": True,
                "reason": f"select supported states {n_select_sup} < "
                          f"{args.min_select_supported_states}; run diagnose/Commit D first",
            }
            continue
        member_out = r2_root / member
        member_out.mkdir(parents=True, exist_ok=True)
        seed_results = []
        for seed in seeds:
            seed_dir = member_out / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            res = run_member_seed(
                cfg=cfg,
                hooks=hooks,
                member=member,
                seed=seed,
                panel_train=panel_train,
                panel_select=panel_select,
                cache_rows=cache_rows,
                out_dir=seed_dir,
                checkpoints=cfg.budget_spec().checkpoints,
                borderline_max_step=cfg.budget_spec().borderline_max_step,
                extend_to_16=bool(args.extend_16),
                regen_limit=int(args.regen_limit),
            )
            seed_results.append(res)
        # Fixed-budget summary at primary budget across seeds (guide §9.4).
        primary = int(cfg.raw.get("primary_budget", 8))
        statuses = []
        for res in seed_results:
            st = (res["stages"].get(primary) or {}).get("verdict", {}).get("status")
            statuses.append(st)
        if statuses and all(s == "PROMOTE" for s in statuses):
            summary_status = "PROMOTE"
        elif any(s in {"GUARD_FAIL", "DEPRIORITIZE"} for s in statuses):
            summary_status = "GUARD_OR_DEPRIORITIZED"
        else:
            summary_status = "BORDERLINE"
        report["members_run"][member] = {
            "skipped": False,
            "n_select_supported_states": n_select_sup,
            "seeds": seed_results,
            "primary_budget": primary,
            "per_seed_status_at_primary": statuses,
            "summary_status": summary_status,
        }
    report["finished_at"] = time.strftime("%F %T")
    (model_dir / "R2_C1_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    summary = {
        m: {
            "summary_status": v.get("summary_status"),
            "per_seed": v.get("per_seed_status_at_primary"),
            "skipped": v.get("skipped", False),
        }
        for m, v in report["members_run"].items()
    }
    (model_dir / "R2_C1_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2000])
    return 0


def states_by_id_split(p: dict[str, Any], states: list[ProbeCState]) -> str:
    for st in states:
        if st.state_id == p["state_id"]:
            return st.split
    return ""


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_member_seed", "main"]
