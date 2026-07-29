"""Analytics for SCOPE v3 formal audit (P/R, premature stop breakdown)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState
from training.audit_scope_chat_online import (
    CAPABILITY_DISPLAY,
    label_local_capabilities,
    label_local_decision,
    summarize_capability_table,
)

_STOP_TYPES = frozenset(
    {
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.ABSTAIN,
    }
)

_REASON_TO_DISPLAY = {
    "DUPLICATE_EVIDENCE": "Duplicate Evidence",
    "PREMATURE_STOP": "Premature Stop",
    "MISSING_DIRECT_EVIDENCE": "Premature Stop",
}


def _parse_state_action(ev: dict[str, Any]) -> tuple[DecisionState | None, CapabilityAction | None]:
    ds = ev.get("decision_state")
    sa = ev.get("student_action_struct") or ev.get("student_action")
    if isinstance(ds, dict) and isinstance(sa, dict):
        try:
            return DecisionState.from_dict(ds), CapabilityAction.from_dict(sa)
        except Exception:
            pass
    return None, None


def enrich_event_local_gt(ev: dict[str, Any]) -> dict[str, Any]:
    """Attach Go25-compatible local GT fields to a v3 event."""
    state, action = _parse_state_action(ev)
    mid = str(ev.get("module_id") or "")
    if state is None or action is None:
        return ev
    local_caps = label_local_capabilities(state, action, mid)
    local_label = label_local_decision(state, action, mid)
    out = dict(ev)
    out["local_capabilities"] = local_caps
    out["local_label"] = local_label
    route = str(ev.get("route", "IGNORE")).lower()
    out["mode"] = {"endorse": "endorse", "correct": "correct"}.get(route, "ignore")
    art = ev.get("artifact") or {}
    out["reason_code"] = art.get("reason_code") or ev.get("reason_code") or ""
    out["action_arguments"] = action.arguments
    out["add_ids"] = list(action.arguments.get("add_ids") or [])
    out["query"] = state.query[:500]
    out["rendered_context"] = (state.rendered_context or "")[:4000]
    out["verification_records"] = [
        {
            "turn_id": r.turn_id,
            "claim": r.claim[:200],
            "document_ids": list(r.document_ids),
            "judgments": dict(r.judgments),
        }
        for r in state.verification_records[-5:]
    ]
    out["pool"] = len(state.pool_document_ids)
    out["curated"] = len(state.curated_document_ids)
    out["remaining_turns"] = state.remaining_turns
    return out


def _bucket_turn(turn: int) -> str:
    if turn <= 5:
        return "turn_1_5"
    if turn <= 15:
        return "turn_6_15"
    if turn <= 25:
        return "turn_16_25"
    return "turn_26_plus"


def _bucket_coverage(n_curated: int) -> str:
    if n_curated <= 0:
        return "curated_0"
    if n_curated <= 2:
        return "curated_1_2"
    if n_curated <= 5:
        return "curated_3_5"
    return "curated_6_plus"


def _bucket_remaining(rem: int) -> str:
    if rem <= 5:
        return "rem_0_5"
    if rem <= 15:
        return "rem_6_15"
    if rem <= 25:
        return "rem_16_25"
    return "rem_26_plus"


def _route_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(str(e.get("route", "IGNORE")).upper() for e in events)
    return {
        "endorse": int(c.get("ENDORSE", 0)),
        "correct": int(c.get("CORRECT", 0)),
        "ignore": int(c.get("IGNORE", 0)),
        "calls": len(events),
    }


def premature_stop_breakdown(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Premature-stop diagnostics with GT buckets."""
    enriched = [enrich_event_local_gt(e) for e in events]

    stop_events = []
    for ev in enriched:
        sa = ev.get("student_action") or {}
        at = sa.get("action_type") if isinstance(sa, dict) else ev.get("student_action")
        if at in {t.value for t in _STOP_TYPES} and ev.get("module_id") == "verification":
            stop_events.append(ev)

    n_stop_total = len(stop_events)
    n_bad_stop = sum(
        1 for e in stop_events if "PREMATURE_STOP" in (e.get("local_capabilities") or [])
        or "MISSING_DIRECT_EVIDENCE" in (e.get("local_capabilities") or [])
    )
    n_valid_stop = sum(
        1
        for e in stop_events
        if e.get("local_label") == "good"
        or (
            "PREMATURE_STOP" not in (e.get("local_capabilities") or [])
            and "MISSING_DIRECT_EVIDENCE" not in (e.get("local_capabilities") or [])
            and any(
                any(bool(v) for v in (r.get("judgments") or {}).values())
                for r in (e.get("verification_records") or [])
            )
        )
    )

    prem_cap = [e for e in enriched if e.get("capability_id") == "premature_stop"]
    routes = _route_counts(prem_cap)

    def _bucket_agg(bucket_name: str, key_fn) -> dict[str, Any]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for e in prem_cap:
            buckets[key_fn(e)].append(e)
        return {
            k: {**_route_counts(v), "n": len(v)}
            for k, v in sorted(buckets.items())
        }

    return {
        "n_stop_total": n_stop_total,
        "n_valid_stop": n_valid_stop,
        "n_bad_stop": n_bad_stop,
        "endorse": routes["endorse"],
        "correct": routes["correct"],
        "ignore": routes["ignore"],
        "calls": routes["calls"],
        "pct_correct_among_premature_cap": (
            routes["correct"] / routes["calls"] if routes["calls"] else 0.0
        ),
        "pct_endorse_among_premature_cap": (
            routes["endorse"] / routes["calls"] if routes["calls"] else 0.0
        ),
        "by_turn": _bucket_agg(
            "turn",
            lambda e: _bucket_turn(int(e.get("turn_id", 0))),
        ),
        "by_evidence_coverage": _bucket_agg(
            "coverage",
            lambda e: _bucket_coverage(int(e.get("curated", 0))),
        ),
        "by_remaining_turns": _bucket_agg(
            "remaining",
            lambda e: _bucket_remaining(int(e.get("remaining_turns", 0))),
        ),
    }


def capability_precision_recall(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = [enrich_event_local_gt(e) for e in events]
    return summarize_capability_table(enriched)


def build_formal_audit_report(
    events: list[dict[str, Any]],
    *,
    n_queries: int,
    base_summary: dict[str, Any],
    go25_reference: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Merge v3 summary with Go25-style P/R and premature breakdown."""
    go25_reference = go25_reference or {
        "Duplicate Evidence": {"precision": 1.0, "recall": 1.0},
        "Premature Stop": {"precision": 1.0, "recall": 0.96},
    }
    table = capability_precision_recall(events)
    by_name = {r["capability"]: r for r in table}
    prem = premature_stop_breakdown(events)

    cap_compare: dict[str, Any] = {}
    for name in ("Duplicate Evidence", "Premature Stop"):
        row = by_name.get(name, {})
        ref = go25_reference.get(name, {})
        cap_compare[name] = {
            "calls": row.get("calls", 0),
            "correct_predictions": row.get("correct", 0),
            "precision": row.get("precision", 0.0),
            "recall": row.get("recall", 0.0),
            "go25_precision": ref.get("precision"),
            "go25_recall": ref.get("recall"),
            "delta_precision": round(
                float(row.get("precision", 0.0)) - float(ref.get("precision", 0.0)), 3
            ),
            "delta_recall": round(
                float(row.get("recall", 0.0)) - float(ref.get("recall", 0.0)),
                3,
            ),
        }

    report = dict(base_summary)
    report["mode"] = base_summary.get("mode", "scope_v3_formal_audit")
    report["n_queries"] = n_queries
    report["capability_precision_recall"] = table
    report["capability_vs_go25"] = cap_compare
    report["PrematureStop"] = prem
    report["training_readiness"] = {
        "dup_healthy": cap_compare.get("Duplicate Evidence", {}).get("precision", 0) >= 0.9,
        "premature_precision_ok": cap_compare.get("Premature Stop", {}).get("precision", 0)
        >= 0.9,
        "premature_all_correct_risk": (
            prem.get("calls", 0) > 0 and prem.get("endorse", 0) == 0
        ),
        "n_trainable_samples": base_summary.get("n_trainable_samples", 0),
        "endorse_correct_ratio": base_summary.get("endorse_correct_ratio", 0.0),
    }
    return report
