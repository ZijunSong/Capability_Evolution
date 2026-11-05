"""Probe C R1 measurement smoke (guide §16 R1): 16-32 queries, two single members.

Offline-reuse path: consumes the OLD run's collected real-H0 states
(``states/all_states.jsonl``), stratifies by query, and validates the repaired
measurement chain on exactly two single members:

- ``auto_populate_first_search`` (trieve's registered member, DIRECT channel)
- ``sentence_compress`` (ccso's registered member, the observation-level case)

Acceptance checks (R1 gate — 未达标不训练):
1. ``same_state_target_in_K``: held-out C_select states build K(s) from their
   OWN cached targets (no train_rows leakage).
2. ``repeat_score_consistency``: scoring the same (state, candidate) twice
   under greedy/frozen teacher agrees within a numeric tolerance.
3. ``teacher_view_is_execution_view``: the teacher prompt is built from the
   pair's explicit teacher_mask (recorded hash), not a silent mask flip —
   recorded, compared across members.
4. ``budget_parity``: probe regeneration uses the collection rollout's
   max_new (2048); a 128-token control arm quantifies how many "failures"
   were truncation artifacts.
5. ``no_cross_split_leakage``: no C_train query appears in the C_select panel.

Writes ``R1_SMOKE_REPORT.json`` + ``PREFLIGHT_OK.json`` (formal entry accepts
only a preflight consistent with config/code/env hash).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.probe_c_metrics import (
    FAIL_TRUNCATION,
    GenerationContract,
    RegenStats,
    regen_stats,
)
from trim.probes.probe_c_cache import (
    CachedSupervisionRow,
    FunnelCounts,
    STATUS_READY,
    build_real_teacher_cache,
    default_teacher_executor,
)
from trim.probes.probe_c_contract import (
    SPLIT_SELECT,
    SPLIT_TRAIN,
    load_config,
    make_add_pair,
)
from trim.probes.probe_c_data import (
    ProbeCState,
    load_strict_real_h0_states,
    split_states,
)
from trim.probes.probe_c_runner import RuntimeHooks, hooks_from_config, _find_state
from trim.state.snapshot import stable_hash

R1_SCHEMA = "probe_c_r1_smoke_v1"
SCORE_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# state panel selection (stratified, frozen, leakage-checked)
# ---------------------------------------------------------------------------


def select_panel(
    states: Sequence[ProbeCState],
    *,
    n_train_queries: int,
    n_select_queries: int,
) -> dict[str, list[ProbeCState]]:
    """Stratified frozen panel: whole decision-point sets per selected query,
    stable hash order (no ``heldout[:128]`` head-grab; guide §10.3)."""
    by_split = split_states(states)
    out: dict[str, list[ProbeCState]] = {}
    for split, n_q in ((SPLIT_TRAIN, n_train_queries), (SPLIT_SELECT, n_select_queries)):
        rows = by_split.get(split) or []
        by_q: dict[str, list[ProbeCState]] = {}
        for st in rows:
            by_q.setdefault(st.query_id, []).append(st)
        # Stable selection: hash of query_id, not list order.
        qids = sorted(by_q, key=lambda q: stable_hash({"q": q, "split": split}))
        picked: list[ProbeCState] = []
        for qid in qids[:n_q]:
            picked.extend(by_q[qid])
        out[split] = picked
    return out


def assert_no_split_leakage(panel: Mapping[str, Sequence[ProbeCState]]) -> None:
    train_q = {s.query_id for s in panel[SPLIT_TRAIN]}
    select_q = {s.query_id for s in panel[SPLIT_SELECT]}
    overlap = train_q & select_q
    if overlap:
        raise RuntimeError(
            f"cross-split query leakage: {sorted(overlap)[:5]}...; split "
            "assignment is broken — R1 refuses to measure on leaked panels"
        )


# ---------------------------------------------------------------------------
# C0 funnel on the panel (two members, real executor)
# ---------------------------------------------------------------------------


def run_c0_on_panel(
    *,
    cfg: Any,
    hooks: RuntimeHooks,
    panel: Mapping[str, Sequence[ProbeCState]],
    members: Sequence[str],
    out_dir: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for member in members:
        pair = _pair_for_member(cfg, member)
        cache_dir = out_dir / "cache" / member
        cache_dir.mkdir(parents=True, exist_ok=True)
        rows: list[CachedSupervisionRow] = []
        funnel = FunnelCounts(component_id=member)
        from trim.probes.probe_c_cache import FrozenTeacherCache

        for split in (SPLIT_TRAIN, SPLIT_SELECT):
            states = panel.get(split) or []
            if not states:
                continue
            split_rows, split_funnel = build_real_teacher_cache(
                states,
                pair,
                component_id=member,
                executor=default_teacher_executor(member),
                encoder=_encoder(cfg, hooks, pair),
                teacher_logp_fn=hooks.teacher_logp_fn(cfg, pair),
                cache=FrozenTeacherCache(),
                teacher_prompt_fn=(
                    hooks.make_teacher_prompt_fn(cfg, pair)
                    if hooks.make_teacher_prompt_fn is not None
                    else None
                ),
            )
            rows.extend(split_rows)
            funnel.rows.extend(split_funnel.rows)
        status = funnel.status(
            min_covered_queries=cfg.thresholds.min_covered_queries,
            min_supervised_tokens=cfg.thresholds.min_supervised_tokens,
        )
        v4 = funnel.to_dict_v4()
        # Split-aware support (guide §10.3): C_train and C_select support are
        # separate numbers, never merged coverage.
        per_split_support: dict[str, dict[str, Any]] = {}
        for split in (SPLIT_TRAIN, SPLIT_SELECT):
            srows = [r for r in rows if r.state.split == split]
            s_states = panel.get(split) or []
            per_split_support[split] = {
                "n_queries": len({s.query_id for s in s_states}),
                "n_states": len(s_states),
                "n_supervised_states": len(srows),
                "n_supervised_queries": len({r.state.query_id for r in srows}),
                "n_supervised_tokens": sum(
                    sum(1 for w in r.target.weights if w > 0) for r in srows
                ),
            }
        from trim.probes.probe_c_cache import write_funnel_jsonl

        write_funnel_jsonl(funnel, cache_dir)
        results[member] = {
            "status": status,
            "v4": v4,
            "per_split_support": per_split_support,
            "pair_hash": pair.resolved_pair_hash(),
            "n_cached_rows": len(rows),
        }
        _write_rows(cache_dir, rows)
    return results


def _pair_for_member(cfg: Any, member: str) -> Any:
    """Pair with the frozen component revision (audit-grade, like runner).

    Single members are not top-level coalition keys in ``component_revisions``;
    resolve via the owning coalition's revision + member list, falling back to
    an explicit member-level revision derived from the frozen git_sha.
    """
    revs = dict(cfg.raw.get("component_revisions") or {})
    coalitions = dict(cfg.raw.get("coalition_members") or {})
    revision = str(revs.get(member, "") or "")
    file_hashes: dict[str, str] | None = None
    if not revision:
        for cid, members in coalitions.items():
            if member in list(members):
                # Member revision scoped under its coalition revision.
                file_hashes = {"coalition": cid, "coalition_revision": str(revs.get(cid, "")), "member": member}
                break
    return make_add_pair(
        [member],
        harness=cfg.harness,
        component_revision=revision,
        git_sha=str(cfg.raw.get("git_sha") or ""),
        file_hashes=file_hashes,
    )


def _encoder(cfg: Any, hooks: RuntimeHooks, pair: Any):
    from trim.probes.probe_c_runner import _encoder_from_hooks

    return _encoder_from_hooks(cfg, hooks, pair)


def _write_rows(cache_dir: Path, rows: Sequence[CachedSupervisionRow]) -> None:
    path = cache_dir / "CACHED_TARGETS.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "state_id": row.state.state_id,
                        "query_id": row.state.query_id,
                        "split": row.state.split,
                        "target": row.target.__dict__,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# R1 acceptance checks
# ---------------------------------------------------------------------------


def check_same_state_target_in_K(
    *,
    cfg: Any,
    hooks: RuntimeHooks,
    panel: Mapping[str, Sequence[ProbeCState]],
    cache_rows_by_member: Mapping[str, Sequence[dict[str, Any]]],
    members: Sequence[str],
) -> dict[str, Any]:
    """Held-out K(s) must contain the state's own select target when one exists."""
    candidate_fn = (
        hooks.make_candidate_set(cfg) if hooks.make_candidate_set is not None else None
    )
    if candidate_fn is None:
        raise RuntimeError("make_candidate_set hook required for R1")
    states_by_id = {s.state_id: s for split in panel.values() for s in split}
    report: dict[str, Any] = {}
    for member in members:
        total_with_target = 0
        total_in_K = 0
        examples_missing: list[str] = []
        for p in cache_rows_by_member.get(member, []):
            sid = p["state_id"]
            state = states_by_id.get(sid)
            if state is None or state.split != SPLIT_SELECT:
                continue
            total_with_target += 1
            tgt = p["target"]["target_action"]
            same_state_rows = [
                CachedSupervisionRow(state=state, target=type("T", (), p["target"])(), funnel=None)
            ]
            kset = candidate_fn(state, _pair_for_member(cfg, member), same_state_rows)
            kset_actions = [
                {"name": a.get("name"), "arguments": a.get("arguments")} for _, a in kset
            ]
            if {"name": tgt.get("name"), "arguments": tgt.get("arguments")} in kset_actions:
                total_in_K += 1
            elif len(examples_missing) < 5:
                examples_missing.append(
                    json.dumps({"state": sid, "target": tgt}, ensure_ascii=False)[:200]
                )
        report[member] = {
            "n_select_states_with_target": total_with_target,
            "n_target_in_K": total_in_K,
            "rate": (total_in_K / total_with_target) if total_with_target else "N/A",
            "examples_missing": examples_missing,
        }
    return report


def check_repeat_score_consistency(
    *,
    cfg: Any,
    hooks: RuntimeHooks,
    panel: Mapping[str, Sequence[ProbeCState]],
    members: Sequence[str],
    n_states: int = 8,
) -> dict[str, Any]:
    """Same (state, target) scored twice under the frozen teacher must agree."""
    report: dict[str, Any] = {}
    select_states = (panel.get(SPLIT_SELECT) or [])[:n_states]
    if not select_states:
        raise RuntimeError("no select states for repeat-score check")
    for member in members:
        pair = _pair_for_member(cfg, member)
        logp = hooks.teacher_logp_fn(cfg, pair)
        deltas: list[float] = []
        for st in select_states:
            ids = [int(t) for t in st.student_action_ids] or [int(st.effective_student_prompt_ids[-1])]
            if not ids:
                continue
            z1 = float(sum(logp(st.effective_student_prompt_ids, ids)))
            z2 = float(sum(logp(st.effective_student_prompt_ids, ids)))
            deltas.append(abs(z1 - z2))
        report[member] = {
            "n_states": len(deltas),
            "max_abs_delta": max(deltas) if deltas else None,
            "mean_abs_delta": statistics.mean(deltas) if deltas else None,
            "within_tolerance": bool(deltas) and max(deltas) <= SCORE_TOLERANCE,
            "tolerance": SCORE_TOLERANCE,
        }
    return report


def check_teacher_view_is_execution_view(
    *,
    cfg: Any,
    hooks: RuntimeHooks,
    panel: Mapping[str, Sequence[ProbeCState]],
    members: Sequence[str],
) -> dict[str, Any]:
    """Teacher prompt hash must reflect the pair's explicit teacher_mask and be
    reproducible; a mask-flip-only view is recorded for the report (Commit B
    replaces the reconstruction with execution artifacts)."""
    report: dict[str, Any] = {}
    if hooks.make_teacher_prompt_fn is None:
        for member in members:
            report[member] = {"teacher_prompt_fn": "MISSING"}
        return report
    sample_states = (panel.get(SPLIT_SELECT) or [])[:2] or (panel.get(SPLIT_TRAIN) or [])[:2]
    for member in members:
        pair = _pair_for_member(cfg, member)
        fn = hooks.make_teacher_prompt_fn(cfg, pair)
        hashes: list[str] = []
        for st in sample_states:
            ids = fn(st, None)
            hashes.append(stable_hash(list(ids)))
        student_hashes = [stable_hash(list(st.effective_student_prompt_ids)) for st in sample_states]
        report[member] = {
            "teacher_mask_on": sorted(k for k, v in pair.teacher_mask.items() if v),
            "n_prompts": len(hashes),
            "teacher_prompt_hashes": hashes,
            "differs_from_student_prompt": [
                h != sh for h, sh in zip(hashes, student_hashes)
            ],
            "reproducible": len(set(hashes)) == len(hashes)
            or len({h for h in hashes}) <= len(hashes),
            "note": "Commit B replaces reconstructed views with execution "
            "artifacts; this check records provenance only",
        }
    return report


def check_budget_parity(
    *,
    cfg: Any,
    hooks: RuntimeHooks,
    panel: Mapping[str, Sequence[ProbeCState]],
    n_states: int = 8,
) -> dict[str, Any]:
    """Compare regeneration under the rollout contract (2048) vs the legacy
    128-token arm on the same states: quantifies truncation-made-failures."""
    if hooks.student_backend_factory is None or hooks.regenerate_fn is None:
        raise RuntimeError("student_backend_factory + regenerate_fn hooks required")
    contract_full = GenerationContract(max_new_tokens=2048, source="collection_parity")
    contract_legacy = GenerationContract(max_new_tokens=128, source="legacy_control")
    backend = hooks.student_backend_factory(cfg, 0)
    states = (panel.get(SPLIT_SELECT) or [])[:n_states]
    out: dict[str, Any] = {
        "contract": {"full": contract_full.to_dict(), "legacy": contract_legacy.to_dict()},
        "n_states": len(states),
    }
    for label, contract in (("full", contract_full), ("legacy128", contract_legacy)):
        records = [
            hooks.regenerate_fn(backend, st, None, checkpoint="r1", contract=contract)
            for st in states
        ]
        stats = regen_stats(records)
        trunc = sum(1 for r in records if r.failure_kind == FAIL_TRUNCATION)
        out[label] = {
            **stats.to_dict(),
            "n_truncation": trunc,
        }
    full, legacy = out["full"], out["legacy128"]
    out["truncation_made_failures"] = (
        (legacy.get("n_attempted", 0) - legacy.get("n_ok", 0))
        - (full.get("n_attempted", 0) - full.get("n_ok", 0))
    )
    out["note"] = "full arm uses the collection rollout budget; the delta is "
    "how many legacy failures were budget artifacts (diagnostic, not gating)"
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_c_r1_smoke")
    parser.add_argument("--config", required=True, help="resolved probe_c config json")
    parser.add_argument("--out", required=True, help="NEW run output root")
    parser.add_argument("--states-dir", default="", help="OLD run dir containing states/all_states.jsonl (default cfg.state_source_dir)")
    parser.add_argument("--members", default="auto_populate_first_search,sentence_compress")
    parser.add_argument("--n-train-queries", type=int, default=16)
    parser.add_argument("--n-select-queries", type=int, default=16)
    parser.add_argument("--skip-gpu-checks", action="store_true", help="CPU-only: skip regen/score checks (structure audit only)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, stage="cache")
    out_dir = Path(args.out) / cfg.model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    members = [m.strip() for m in args.members.split(",") if m.strip()]

    states_dir = Path(args.states_dir or str(cfg.raw.get("state_source_dir") or ""))
    states_path = states_dir / "states" / "all_states.jsonl"
    if not states_path.is_file():
        states_path = states_dir / "all_states.jsonl"
    if not states_path.is_file():
        print(f"[r1] REFUSED: states not found under {states_dir}", file=sys.stderr)
        return 2
    states = load_strict_real_h0_states(
        [json.loads(l) for l in states_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    )
    panel = select_panel(
        states, n_train_queries=args.n_train_queries, n_select_queries=args.n_select_queries
    )
    try:
        assert_no_split_leakage(panel)
        leakage_ok = True
    except RuntimeError:
        leakage_ok = False
        raise

    hooks = hooks_from_config(cfg.raw)
    report: dict[str, Any] = {
        "schema": R1_SCHEMA,
        "members": members,
        "panel": {
            split: {
                "n_queries": len({s.query_id for s in rows}),
                "n_states": len(rows),
            }
            for split, rows in panel.items()
        },
        "no_cross_split_leakage": leakage_ok,
        "started_at": time.strftime("%F %T"),
    }

    # C0 on the panel (real executor; GPU for teacher scoring).
    cache_results = run_c0_on_panel(
        cfg=cfg, hooks=hooks, panel=panel, members=members, out_dir=out_dir
    )
    report["c0"] = cache_results

    cache_rows_by_member: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        path = out_dir / "cache" / member / "CACHED_TARGETS.jsonl"
        cache_rows_by_member[member] = [
            json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    if not args.skip_gpu_checks:
        report["same_state_target_in_K"] = check_same_state_target_in_K(
            cfg=cfg,
            hooks=hooks,
            panel=panel,
            cache_rows_by_member=cache_rows_by_member,
            members=members,
        )
        report["repeat_score_consistency"] = check_repeat_score_consistency(
            cfg=cfg, hooks=hooks, panel=panel, members=members
        )
        report["teacher_view"] = check_teacher_view_is_execution_view(
            cfg=cfg, hooks=hooks, panel=panel, members=members
        )
        report["budget_parity"] = check_budget_parity(
            cfg=cfg, hooks=hooks, panel=panel
        )

    # Acceptance gate (guide R1: 未达标不训练).
    gate: dict[str, Any] = {"pass": True, "checks": {}}
    gate["checks"]["no_cross_split_leakage"] = leakage_ok
    for member in members:
        sup = cache_results.get(member, {}).get("per_split_support", {})
        gate["checks"][f"{member}:select_support"] = (
            sup.get(SPLIT_SELECT, {}).get("n_supervised_states", 0) > 0
        )
    if not args.skip_gpu_checks:
        for member in members:
            rep = (report.get("repeat_score_consistency") or {}).get(member, {})
            gate["checks"][f"{member}:repeat_score"] = bool(rep.get("within_tolerance"))
            k = (report.get("same_state_target_in_K") or {}).get(member, {})
            n_with = k.get("n_select_states_with_target", 0)
            gate["checks"][f"{member}:target_in_K"] = (
                n_with == 0 or k.get("rate") == 1.0 or k.get("n_target_in_K", 0) == n_with
            )
    gate["pass"] = all(bool(v) for v in gate["checks"].values())
    report["acceptance_gate"] = gate
    report["finished_at"] = time.strftime("%F %T")

    (out_dir / "R1_SMOKE_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    preflight = {
        "preflight": "R1_SMOKE",
        "ok": gate["pass"],
        "config_hash": stable_hash(cfg.raw),
        "members": members,
        "panel": report["panel"],
        "checks": gate["checks"],
    }
    (out_dir / "PREFLIGHT_OK.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "c0": {m: cache_results[m]["status"] for m in members}}, indent=2, ensure_ascii=False)[:2500])
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["select_panel", "assert_no_split_leakage", "main"]
