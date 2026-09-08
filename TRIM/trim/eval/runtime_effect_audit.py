"""Fail-closed audit: a harness mask must change live env behavior.

Config JSON is not evidence. Each ON bit must produce a detectable delta
versus the same action under zero mask. Each OFF bit must stay inert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trim.adapters.components import full_mask, zero_mask
from trim.eval.local_search_env import execute_tool, new_state, wm_text

CLAIM_NOT_USABLE = "NOT_USABLE_FOR_FULL_VS_ZERO"
CLAIM_WIRING_OK = "RUNTIME_WIRING_OK"
LIVE_FIRE_THRESHOLD = 1.0


_PROBE_STORE = {
    "d1": {
        "id": "d1",
        "text": "Alice Smith visited Paris in 2019. The treaty named Bob Jones. "
        * 8,
    },
    "d2": {
        "id": "d2",
        "text": "Alice Smith later joined Carol Adams in Paris. The 2019 treaty held.",
    },
    "d3": {
        "id": "d3",
        "text": "Alice Smith visited Paris in 2019. The treaty named Bob Jones. "
        * 8,
    },
}


def _search(state: dict[str, Any]) -> tuple[dict[str, Any], str, bool]:
    return execute_tool(state, "search_corpus", {"query": "Alice Smith Paris 2019"})


def _probe_auto_populate(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, obs, ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    seeded = list(st.get("auto_seed") or [])
    return {
        "ok": bool(ok),
        "n_curated": len(st.get("curated") or {}),
        "auto_seed": seeded,
        "obs_auto": "[AUTO]" in obs,
        "wm_auto_on": "auto_populate_first_search=ON" in wm_text(st),
        "fired": bool(seeded) and bool((st.get("runtime_effects") or {}).get("auto_populate_first_search")),
    }


def _probe_verify(mask: Mapping[str, bool]) -> dict[str, Any]:
    st = new_state("Alice Smith", dict(_PROBE_STORE), harness_mask=mask)
    st, _obs, search_ok = _search(st)
    st2, obs, ok = execute_tool(st, "verify", {"doc_ids": ["d1"], "claim": "Alice visited Paris"})
    return {
        "search_ok": bool(search_ok),
        "legal": bool(ok),
        "obs_error": obs.startswith("ERROR: invalid tool"),
        "fired": bool((st2.get("runtime_effects") or {}).get("verify_tool")),
    }


def _probe_importance(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, _obs, _ok = _search(new_state("Alice", dict(_PROBE_STORE), harness_mask=mask))
    # Clear auto-curated so curate is the only writer.
    st["curated"] = {}
    st["importance"] = {}
    st["auto_seed"] = None
    st, _obs, ok = execute_tool(
        st,
        "curate",
        {"add_ids": ["d1"], "importance": {"d1": "very_high"}},
    )
    stored = (st.get("importance") or {}).get("d1")
    wm = wm_text(st)
    return {
        "ok": bool(ok),
        "stored": stored,
        "wm_has_tag": "importance=very_high" in wm,
        "fired": stored == "very_high",
    }


def _probe_subtractive(mask: Mapping[str, bool]) -> dict[str, Any]:
    from trim.eval.h1_component_runtime import MAX_CURATED_DOCS

    store = {f"c{i}": {"id": f"c{i}", "text": f"Doc {i} Alice Smith", "score": 1.0} for i in range(MAX_CURATED_DOCS + 2)}
    st = new_state("Alice", store, harness_mask=mask)
    st["pool"] = dict(store)
    st["first_search_pending"] = False
    for i in range(MAX_CURATED_DOCS):
        st, _obs, _ok = execute_tool(
            st,
            "curate",
            {"add_ids": [f"c{i}"], "importance": {f"c{i}": "low"}},
        )
    before = set(st.get("curated") or {})
    st, obs, ok = execute_tool(
        st,
        "curate",
        {"add_ids": [f"c{MAX_CURATED_DOCS}"], "importance": {f"c{MAX_CURATED_DOCS}": "very_high"}},
    )
    after = set(st.get("curated") or {})
    evicted = bool("[EVICTED" in obs) or (before - after)
    added = f"c{MAX_CURATED_DOCS}" in after
    return {
        "ok": bool(ok),
        "evicted": bool(evicted),
        "added_over_cap": added,
        "fired": bool(evicted) and added,
    }


def _probe_evidence_graph(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, obs, ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    wm = wm_text(st)
    has = "[Evidence Graph]" in obs or "[Evidence Graph]" in wm
    return {"ok": bool(ok), "visible": has, "fired": has}


def _probe_sentence_compress(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, obs, ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    fired = bool((st.get("runtime_effects") or {}).get("sentence_compress"))
    return {"ok": bool(ok), "obs_len": len(obs), "fired": fired}


def _probe_content_dedup(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, _obs, ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    dropped = list(st.get("dedup_dropped") or [])
    fired = bool(dropped) or bool((st.get("runtime_effects") or {}).get("content_dedup"))
    return {"ok": bool(ok), "dropped": dropped, "n_pool": len(st.get("pool") or {}), "fired": fired}


def _probe_token_budget(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, obs, ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    has = "[Context:" in obs or bool(st.get("token_budget_marker"))
    return {"ok": bool(ok), "visible": has, "fired": has}


def _probe_chunk_neighbors(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, _obs, _ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    st, obs, ok = execute_tool(st, "read_document", {"doc_id": "d1"})
    has = "[Chunk neighbors]" in obs
    return {"ok": bool(ok), "visible": has, "fired": has}


def _probe_rerank(mask: Mapping[str, bool]) -> dict[str, Any]:
    st, obs, ok = _search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=mask))
    has = "[Rerank instruction]" in obs or bool(st.get("rerank_instruction"))
    return {"ok": bool(ok), "visible": has, "fired": has}


PROBES = {
    "auto_populate_first_search": _probe_auto_populate,
    "verify_tool": _probe_verify,
    "importance_tagging": _probe_importance,
    "subtractive_curation": _probe_subtractive,
    "evidence_graph": _probe_evidence_graph,
    "sentence_compress": _probe_sentence_compress,
    "content_dedup": _probe_content_dedup,
    "token_budget_marker": _probe_token_budget,
    "chunk_neighbors": _probe_chunk_neighbors,
    "adaptive_rerank_instruction": _probe_rerank,
}


def _expected_on(mask: Mapping[str, bool], cid: str) -> bool:
    return bool(mask.get(cid, False))


def audit_mask_wiring(mask: Mapping[str, bool] | None) -> dict[str, Any]:
    """Deterministic full-vs-zero probe. Independent of the policy model."""
    mask = dict(mask or zero_mask())
    zero = zero_mask()
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for cid, probe in PROBES.items():
        on_report = probe(mask)
        off_report = probe(zero)
        want_on = _expected_on(mask, cid)
        on_fired = bool(on_report.get("fired"))
        off_fired = bool(off_report.get("fired"))
        if want_on and not on_fired:
            failures.append(f"{cid}: mask ON but probe did not fire")
        if want_on and on_fired and off_fired and on_report == off_report:
            failures.append(f"{cid}: ON and zero probes are identical")
        if not want_on and on_fired:
            failures.append(f"{cid}: mask OFF but probe fired")
        if off_fired:
            failures.append(f"{cid}: zero mask must stay inert")
        rows[cid] = {
            "mask_on": want_on,
            "full_or_current": on_report,
            "zero": off_report,
            "pass": (on_fired if want_on else not on_fired) and not off_fired,
        }
    passed = not failures
    n_on = sum(1 for v in mask.values() if v)
    return {
        "pass": passed,
        "claim_usable_for_full_vs_zero": False if not passed else None,
        "n_on": n_on,
        "n_fail": len(failures),
        "failures": failures,
        "components": rows,
        "mask": dict(mask),
        "gate": "wiring_probe",
        "threshold": {
            "on_must_fire": True,
            "zero_must_be_inert": True,
            "live_fire_rate": LIVE_FIRE_THRESHOLD,
        },
        "summary": (
            "Harness-1 mask wiring probe passed."
            if passed
            else "Harness-1 mask wiring probe FAILED: " + "; ".join(failures)
        ),
    }


def summarize_live_effects(
    traces: list[Mapping[str, Any]],
    mask: Mapping[str, bool] | None,
) -> dict[str, Any]:
    mask = dict(mask or {})
    n = len(traces)
    failures: list[str] = []
    rates: dict[str, Any] = {}
    for cid, on in mask.items():
        fires = 0
        opportunities = 0
        for tr in traces:
            effects = tr.get("runtime_effects") or {}
            n_search = float(tr.get("n_search_calls") or 0.0)
            if cid == "auto_populate_first_search":
                names = [str(x) for x in (tr.get("tool_names") or tr.get("names") or [])]
                first = next((n for n in names if n and n != "unknown"), "")
                if first not in {"search_corpus", "grep_corpus", "fan_out_search"}:
                    continue
                opportunities += 1
                if int(effects.get(cid) or 0) > 0 or tr.get("auto_seed"):
                    fires += 1
            elif cid == "verify_tool":
                opportunities += 1
                if int(effects.get(cid) or 0) > 0:
                    fires += 1
            else:
                if n_search <= 0 and cid in {
                    "evidence_graph",
                    "sentence_compress",
                    "content_dedup",
                    "token_budget_marker",
                    "adaptive_rerank_instruction",
                    "chunk_neighbors",
                }:
                    continue
                opportunities += 1
                if int(effects.get(cid) or 0) > 0:
                    fires += 1
        rate = (fires / opportunities) if opportunities else None
        rates[cid] = {"on": bool(on), "fires": fires, "opportunities": opportunities, "rate": rate}
        if on and cid == "auto_populate_first_search" and opportunities and (rate or 0.0) < LIVE_FIRE_THRESHOLD:
            failures.append(
                f"{cid}: live fire rate {rate:.3f} < {LIVE_FIRE_THRESHOLD} "
                f"({fires}/{opportunities} searched queries)"
            )
        if not on and fires:
            failures.append(f"{cid}: mask OFF but live effects fired {fires} times")
    return {
        "pass": not failures,
        "n_traces": n,
        "rates": rates,
        "failures": failures,
        "gate": "live_effects",
        "threshold": {"auto_populate_fire_rate": LIVE_FIRE_THRESHOLD},
    }


def merge_audits(wiring: dict[str, Any], live: dict[str, Any] | None = None) -> dict[str, Any]:
    failures = list(wiring.get("failures") or [])
    if live:
        failures.extend(live.get("failures") or [])
    passed = not failures and bool(wiring.get("pass")) and (live is None or bool(live.get("pass")))
    n_on = int(wiring.get("n_on") or 0)
    usable = False
    reason = "wiring or live effect gate failed"
    if passed and n_on == 0:
        reason = (
            "zero-mask run is valid H_min telemetry only; a paired full-mask run "
            "that also passes this audit is required before any full-vs-zero claim"
        )
    elif passed and n_on > 0:
        usable = False
        reason = (
            "this single run passed wiring/live gates, but a full-vs-zero claim "
            "still needs a paired zero-mask run that also passes the same audit"
        )
    if not passed:
        usable = False
        reason = "RUNTIME_EFFECT_AUDIT_FAILED: " + "; ".join(failures)
    return {
        "pass": passed,
        "claim_usable_for_full_vs_zero": usable,
        "claim_status": CLAIM_WIRING_OK if passed else CLAIM_NOT_USABLE,
        "reason": reason,
        "wiring": wiring,
        "live": live,
        "failures": failures,
        "summary": reason if not passed else wiring.get("summary"),
    }


def write_runtime_audit(out: Path, audit: Mapping[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "RUNTIME_EFFECT_AUDIT.json"
    path.write_text(json.dumps(dict(audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def audit_harness_mask_or_raise(mask: Mapping[str, bool] | None, *, out: Path | None = None) -> dict[str, Any]:
    wiring = audit_mask_wiring(mask)
    audit = merge_audits(wiring)
    if out is not None:
        write_runtime_audit(out, audit)
    if not wiring.get("pass"):
        raise RuntimeError(wiring.get("summary") or "mask wiring probe failed")
    return audit


def _opd_projection_probe(
    component_id: str,
    *,
    harness: str | None,
    student_mask: Mapping[str, bool],
) -> dict[str, Any]:
    from trim.training.four_cell_runtime import snap_from_state, teacher_for
    from trim.training.opd_dataset import project_and_materialize
    from trim.training.opd_projection import StudentActionSpaceProjector
    from trim.training.rl_opd_types import StudentDecisionPoint

    st = new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=dict(student_mask))
    st, _obs, search_ok = _search(st)
    snap = snap_from_state("train_probe", st, component_id, harness_mask=dict(student_mask))
    fn = teacher_for(component_id, harness=harness)
    point = StudentDecisionPoint(
        episode_id="train_probe",
        query_id="train_probe",
        rollout_idx=0,
        turn_id=0,
        policy_version="probe",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=[],
        student_action_text="",
        action_tool_names=[],
        post_action_snapshot=snap,
        structurally_valid=True,
    )
    events = fn(point) if fn is not None else []
    projection, steps = project_and_materialize(
        student_snapshot=snap,
        teacher_events=events,
        student_mask=snap.harness_mask,
        component_id=component_id,
        projector=StudentActionSpaceProjector(),
    )
    return {
        "pass": len(steps) >= 1,
        "n_teacher_events": len(events),
        "n_projected_steps": len(steps),
        "projection_kind": getattr(projection.kind, "value", str(projection.kind)),
        "reject_reason": projection.reject_reason,
        "search_ok": bool(search_ok),
        "n_pool": len(st.get("pool") or {}),
        "auto_seed": list(st.get("auto_seed") or []),
        "student_mask": dict(student_mask),
    }


def audit_train_runtime_or_raise(args: Any, *, out: Path | None = None) -> dict[str, Any]:
    """Fail-closed train start: student/teacher masks must wire, TRIM must project."""
    from trim.adapters.harness_profiles import infer_harness_from_ids, is_harness_g
    from trim.training.four_cell_runtime import student_mask_for, teacher_mask_for
    from trim.training.rl_opd_types import TRAINING_MODE_RL, uses_sampled_opd

    component_id = getattr(args, "component", None)
    harness = getattr(args, "harness", None) or infer_harness_from_ids(component_id)
    mode = str(getattr(args, "training_mode", "") or "")
    if is_harness_g(harness) or is_harness_g(component_ids=component_id):
        payload = {
            "pass": True,
            "skipped": True,
            "harness": harness,
            "training_mode": mode,
            "reason": "Harness-G train runtime uses a different env; H1 wiring probe skipped",
        }
        if out is not None:
            write_runtime_audit(out, payload)
        return payload

    student = student_mask_for(component_id, harness=harness)
    teacher = teacher_mask_for(component_id, harness=harness)
    student_w = audit_mask_wiring(student)
    teacher_w = audit_mask_wiring(teacher)
    failures: list[str] = []
    if not student_w.get("pass"):
        failures.extend([f"student: {x}" for x in (student_w.get("failures") or ["wiring failed"])])
    if not teacher_w.get("pass"):
        failures.extend([f"teacher: {x}" for x in (teacher_w.get("failures") or ["wiring failed"])])

    lam = 0.0 if mode == TRAINING_MODE_RL else float(getattr(args, "lambda_opd", 0.0) or 0.0)
    opd_loss = str(getattr(args, "opd_loss", "") or "")
    want_projection = lam > 0.0 and not uses_sampled_opd(opd_loss)
    opd: dict[str, Any] | None = None
    if want_projection:
        opd = _opd_projection_probe(str(component_id), harness=harness, student_mask=student)
        if not opd.get("pass"):
            failures.append(
                "TRIM/OPD projected 0 student-legal steps from the teacher side-branch "
                f"(kind={opd.get('projection_kind')} reject={opd.get('reject_reason')})"
            )

    passed = not failures
    payload = {
        "pass": passed,
        "skipped": False,
        "harness": harness,
        "training_mode": mode,
        "lambda_opd": lam,
        "opd_loss": opd_loss,
        "student_n_on": int(student_w.get("n_on") or 0),
        "teacher_n_on": int(teacher_w.get("n_on") or 0),
        "opd_n_projected_steps": None if opd is None else int(opd.get("n_projected_steps") or 0),
        "student": merge_audits(student_w),
        "teacher": merge_audits(teacher_w),
        "opd_projection": opd,
        "failures": failures,
        "claim_usable_for_full_vs_zero": False,
        "summary": (
            "train runtime wiring probe passed"
            if passed
            else "TRAIN_RUNTIME_AUDIT_FAILED: " + "; ".join(failures)
        ),
    }
    if out is not None:
        write_runtime_audit(out, payload)
    if not passed:
        raise RuntimeError(payload["summary"])
    return payload
