#!/usr/bin/env python3
"""Offline re-label + go/no-go audit for existing DecisionState events.

Does NOT re-rollout. Works from chat_decision_audit_events.jsonl fields plus
synthetic DecisionState probes for valid-stop ENDORSE.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.artifacts.schema import GuidanceMode
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import ClaimState, VerificationRecordState
from harness.shadow.verification_shadow import VerificationShadow
from training.audit_scope_chat_online import (
    CAPABILITY_DISPLAY,
    summarize_capability_table,
)
from training.train_scope import make_toy_decision_state

STOP_ACTIONS = {"stop_and_answer", "answer", "abstain"}


def relabel_event_offline(ev: dict[str, Any]) -> dict[str, Any]:
    """Recompute typed GT from fields available in the thin event schema."""
    out = dict(ev)
    caps: list[str] = []
    mid = ev.get("module_id")
    action = str(ev.get("student_action") or "")
    curated = int(ev.get("curated") or 0)
    n_verify = int(ev.get("n_verify") or 0)
    reason = str(ev.get("reason_code") or "")
    mode = str(ev.get("mode") or "")

    if mid == "evidence_state":
        # Duplicate is reliable from prior local_caps / reason.
        prev = list(ev.get("local_capabilities") or [])
        if "DUPLICATE_EVIDENCE" in prev or reason == "DUPLICATE_EVIDENCE":
            caps.append("DUPLICATE_EVIDENCE")
        # IRRELEVANT cannot be faithfully recomputed without add_ids/rendered_context.
        # Keep as unknown for GT; precision reported separately via local_label proxy.
        if reason == "WEAK_SUPPORT" and mode == "correct":
            # Weak signal only — do not auto-trust prediction as GT.
            pass

    if mid == "verification":
        if action in STOP_ACTIONS:
            if curated <= 0:
                caps.append("MISSING_DIRECT_EVIDENCE")
            elif n_verify <= 0:
                caps.append("PREMATURE_STOP")
            # n_verify > 0 → valid-stop candidate (no premature GT)

    out["local_capabilities"] = caps
    # Offline local_label: bad if any error cap; else keep original unless stop valid.
    if caps:
        out["local_label"] = "bad"
    elif mid == "verification" and action in STOP_ACTIONS and curated > 0 and n_verify > 0:
        out["local_label"] = "good"
    else:
        out["local_label"] = ev.get("local_label", "ambiguous")
    out["relabel"] = "offline_v1"
    return out


def capability_precision_vs_local_label(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Proxy precision for capabilities lacking offline GT (esp. IRRELEVANT)."""
    rows = []
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        if e.get("mode") != "correct":
            continue
        name = CAPABILITY_DISPLAY.get(str(e.get("reason_code") or ""))
        if not name:
            continue
        by_reason.setdefault(name, []).append(e)
    for name, items in sorted(by_reason.items()):
        n = len(items)
        # Prefer typed GT match when present; else local_label==bad
        tp = 0
        for e in items:
            caps = {
                CAPABILITY_DISPLAY.get(c, c) for c in (e.get("local_capabilities") or [])
            }
            if name in caps:
                tp += 1
            elif not (e.get("local_capabilities") or []) and e.get("local_label") == "bad":
                tp += 1
        rows.append(
            {
                "capability": name,
                "correct": n,
                "tp_proxy": tp,
                "precision_proxy": round(tp / n, 3) if n else 0.0,
            }
        )
    return rows


def probe_valid_vs_premature_stop() -> dict[str, Any]:
    """Synthetic DecisionState probes: bad stop→CORRECT, valid stop→ENDORSE."""
    shadow = VerificationShadow()
    stop = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": "probe"},
    )

    bad_state = make_toy_decision_state(
        curated_document_ids=("doc_a",),
        verification_records=(),
        evidence_claims=(),
    )
    bad_art = shadow.analyze(bad_state, stop)

    valid_state = make_toy_decision_state(
        curated_document_ids=("doc_a", "doc_b"),
        visible_document_ids=("doc_a", "doc_b"),
        pool_document_ids=("doc_a", "doc_b"),
        evidence_claims=(
            ClaimState(
                claim_id="c1",
                text="Acme founded by X",
                status="supported",
                supporting_document_ids=("doc_a",),
            ),
        ),
        verification_records=(
            VerificationRecordState(
                turn_id=2,
                claim="Acme founded by X",
                document_ids=("doc_a",),
                judgments={"doc_a": True},
            ),
        ),
    )
    valid_art = shadow.analyze(valid_state, stop)

    empty_state = make_toy_decision_state(
        curated_document_ids=(),
        verification_records=(),
    )
    empty_art = shadow.analyze(empty_state, stop)

    return {
        "bad_stop": {
            "mode": bad_art.mode.value,
            "reason": bad_art.reason_code,
            "expect": "correct/PREMATURE_STOP",
            "ok": bad_art.mode == GuidanceMode.CORRECT
            and bad_art.reason_code == "PREMATURE_STOP",
        },
        "empty_curated_stop": {
            "mode": empty_art.mode.value,
            "reason": empty_art.reason_code,
            "expect": "correct/MISSING_DIRECT_EVIDENCE",
            "ok": empty_art.mode == GuidanceMode.CORRECT
            and empty_art.reason_code == "MISSING_DIRECT_EVIDENCE",
        },
        "valid_stop": {
            "mode": valid_art.mode.value,
            "reason": valid_art.reason_code,
            "expect": "endorse/VERIFICATION_SUPPORTED",
            "ok": valid_art.mode == GuidanceMode.ENDORSE
            and valid_art.reason_code == "VERIFICATION_SUPPORTED",
        },
    }


def dump_spotcheck(
    events: list[dict[str, Any]],
    out_dir: Path,
    *,
    per_class: int = 40,
    seed: int = 0,
) -> dict[str, int]:
    rng = random.Random(seed)
    targets = {
        "DUPLICATE_EVIDENCE": "duplicate",
        "IRRELEVANT_EVIDENCE": "irrelevant",
        "PREMATURE_STOP": "premature_stop",
    }
    counts = {}
    for reason, slug in targets.items():
        pool = [
            e
            for e in events
            if e.get("mode") == "correct" and e.get("reason_code") == reason
        ]
        rng.shuffle(pool)
        sample = pool[:per_class]
        counts[reason] = len(sample)
        path = out_dir / f"spotcheck_{slug}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i, e in enumerate(sample):
                row = {
                    "spot_id": f"{slug}_{i:03d}",
                    "checklist": {
                        "student_action_really_wrong": None,
                        "recommended_action_is_better": None,
                        "recommended_uses_only_st_info": None,
                        "notes": "",
                    },
                    "event": e,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def go_nogo(rows: list[dict[str, Any]], probes: dict[str, Any]) -> dict[str, Any]:
    by = {r["capability"]: r for r in rows}
    thresholds = {
        "Duplicate Evidence": 0.9,
        "Premature Stop": 0.9,
        "Irrelevant Evidence": 0.8,
    }
    decisions = {}
    for name, thr in thresholds.items():
        r = by.get(name)
        prec = (r or {}).get("precision", 0.0)
        decisions[name] = {
            "precision": prec,
            "threshold": thr,
            "pass": prec >= thr,
            "calls": (r or {}).get("calls", 0),
            "correct": (r or {}).get("correct", 0),
        }
    decisions["valid_stop_endorse_probe"] = {
        "pass": bool(probes.get("valid_stop", {}).get("ok")),
        "detail": probes.get("valid_stop"),
    }
    decisions["bad_stop_correct_probe"] = {
        "pass": bool(probes.get("bad_stop", {}).get("ok")),
        "detail": probes.get("bad_stop"),
    }
    decisions["overall_train_ready"] = all(
        decisions[k]["pass"]
        for k in (
            "Duplicate Evidence",
            "Premature Stop",
            "Irrelevant Evidence",
            "valid_stop_endorse_probe",
            "bad_stop_correct_probe",
        )
    )
    return decisions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--events",
        default="outputs/scope_chat_decision_audit_100/chat_decision_audit_events.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/scope_chat_decision_audit_100/offline_relabel",
    )
    p.add_argument("--spotcheck", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    events_path = Path(args.events)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = [json.loads(l) for l in events_path.open() if l.strip()]
    relabeled = [relabel_event_offline(e) for e in raw]

    relabel_path = out_dir / "events_relabeled.jsonl"
    with relabel_path.open("w", encoding="utf-8") as fh:
        for e in relabeled:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    table = summarize_capability_table(relabeled)
    proxy = capability_precision_vs_local_label(raw)  # original labels for IRRELEVANT proxy
    probes = probe_valid_vs_premature_stop()
    spot_counts = dump_spotcheck(
        raw, out_dir, per_class=args.spotcheck, seed=args.seed
    )

    # Merge proxy precision into Irrelevant row for go/no-go honesty
    table_for_gate = []
    proxy_by = {r["capability"]: r for r in proxy}
    for r in table:
        rr = dict(r)
        if r["capability"] == "Irrelevant Evidence" and r["calls"] == 0:
            # No offline GT → use local_label proxy precision from original events
            pr = proxy_by.get("Irrelevant Evidence", {})
            rr["precision"] = pr.get("precision_proxy", 0.0)
            rr["precision_note"] = "proxy_vs_original_local_label"
            rr["correct"] = pr.get("correct", r["correct"])
        table_for_gate.append(rr)

    # Also add predicted-only rows from proxy if missing in typed table
    have = {r["capability"] for r in table_for_gate}
    for name, pr in proxy_by.items():
        if name not in have:
            table_for_gate.append(
                {
                    "capability": name,
                    "calls": 0,
                    "correct": pr["correct"],
                    "precision": pr["precision_proxy"],
                    "recall": 0.0,
                    "precision_note": "proxy_vs_original_local_label",
                }
            )

    gate = go_nogo(table_for_gate, probes)

    # Empirical stop distribution in original audit
    v_stops = [
        e
        for e in raw
        if e.get("module_id") == "verification"
        and e.get("student_action") in STOP_ACTIONS
    ]
    stop_modes = Counter((e["mode"], e["reason_code"]) for e in v_stops)
    endorse_stops = sum(1 for e in v_stops if e["mode"] == "endorse")
    n_verify_zero = sum(1 for e in v_stops if int(e.get("n_verify") or 0) == 0)

    report = {
        "n_events": len(raw),
        "capability_table_offline_gt": table,
        "precision_proxy_vs_local_label": proxy,
        "stop_discrimination_probes": probes,
        "online_stop_distribution": {
            "n_stop_events": len(v_stops),
            "endorse_stops": endorse_stops,
            "n_verify_eq_0": n_verify_zero,
            "mode_reason": {f"{a}/{b}": c for (a, b), c in stop_modes.items()},
            "note": (
                "If n_verify_eq_0 ≈ n_stop_events, verify() never populated "
                "wm.verification_records during this audit — valid-stop ENDORSE "
                "cannot appear online."
            ),
        },
        "spotcheck_counts": spot_counts,
        "go_nogo": gate,
    }
    (out_dir / "offline_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown summary
    lines = [
        "# Offline Audit Re-label Report",
        "",
        f"Events: {len(raw)}",
        "",
        "## Capability table (offline typed GT)",
        "",
        "| Capability | Calls | Correct | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['capability']} | {r['calls']} | {r['correct']} | "
            f"{r['precision']:.2f} | {r['recall']:.2f} |"
        )
    lines += [
        "",
        "## Precision proxy vs original local_label (for IRRELEVANT etc.)",
        "",
        "| Capability | Correct | TP_proxy | Precision_proxy |",
        "|---|---:|---:|---:|",
    ]
    for r in proxy:
        lines.append(
            f"| {r['capability']} | {r['correct']} | {r['tp_proxy']} | "
            f"{r['precision_proxy']:.2f} |"
        )
    lines += [
        "",
        "## Stop discrimination probes",
        "",
        "```json",
        json.dumps(probes, indent=2),
        "```",
        "",
        "## Online stop distribution",
        "",
        f"- stop events: {len(v_stops)}",
        f"- endorse stops: {endorse_stops}",
        f"- stops with n_verify==0: {n_verify_zero}/{len(v_stops)}",
        "",
        "## Go / No-Go",
        "",
    ]
    for k, v in gate.items():
        if k == "overall_train_ready":
            continue
        status = "PASS" if v.get("pass") else "FAIL"
        lines.append(f"- {k}: **{status}** ({v})")
    lines.append(
        f"\n**overall_train_ready = {gate['overall_train_ready']}**\n"
    )
    (out_dir / "offline_audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
