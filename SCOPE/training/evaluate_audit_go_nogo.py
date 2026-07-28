#!/usr/bin/env python3
"""Evaluate go/no-go gates for a fresh DecisionState audit run."""

from __future__ import annotations

import argparse
import json
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
from training.audit_scope_chat_online import CAPABILITY_DISPLAY, summarize_capability_table
from training.train_scope import make_toy_decision_state

STOP = {"stop_and_answer", "answer", "abstain"}


def _prec_recall_table(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_capability_table(events)


def _stop_mix(events: list[dict[str, Any]]) -> dict[str, Any]:
    stops = [
        e
        for e in events
        if e.get("module_id") == "verification" and e.get("student_action") in STOP
    ]
    n = len(stops)
    n_correct = sum(1 for e in stops if e.get("mode") == "correct")
    n_endorse = sum(1 for e in stops if e.get("mode") == "endorse")
    n_verify_pos = sum(
        1
        for e in stops
        if any(
            any(bool(v) for v in (r.get("judgments") or {}).values())
            for r in (e.get("verification_records") or [])
        )
    )
    p_correct = n_correct / n if n else 0.0
    return {
        "n_stops": n,
        "n_correct": n_correct,
        "n_endorse": n_endorse,
        "n_with_positive_verify": n_verify_pos,
        "P_CORRECT_given_stop": round(p_correct, 4),
        "mode_reason": {
            f"{a}/{b}": c for (a, b), c in Counter((e.get("mode"), e.get("reason_code")) for e in stops).items()
        },
    }


def _targeted_valid_stop_probe(n: int = 8) -> dict[str, Any]:
    """Construct evidence-sufficient stop states; expect ENDORSE."""
    shadow = VerificationShadow()
    results = []
    for i in range(n):
        doc = f"doc_{i}"
        state = make_toy_decision_state(
            turn_id=10 + i,
            curated_document_ids=(doc,),
            visible_document_ids=(doc,),
            pool_document_ids=(doc, f"pool_{i}"),
            evidence_claims=(
                ClaimState(
                    claim_id=f"c{i}",
                    text=f"claim text {i}",
                    status="supported",
                    supporting_document_ids=(doc,),
                ),
            ),
            verification_records=(
                VerificationRecordState(
                    turn_id=5 + i,
                    claim=f"claim text {i}",
                    document_ids=(doc,),
                    judgments={doc: True},
                ),
            ),
        )
        action = CapabilityAction(
            action_type=CapabilityActionType.STOP_AND_ANSWER,
            arguments={"reasoning": "evidence sufficient", "targeted": True},
        )
        art = shadow.analyze(state, action)
        results.append(
            {
                "i": i,
                "mode": art.mode.value,
                "reason": art.reason_code,
                "ok": art.mode == GuidanceMode.ENDORSE,
            }
        )
    n_ok = sum(1 for r in results if r["ok"])
    return {"n": n, "n_endorse": n_ok, "pass": n_ok >= min(5, n), "rows": results}


def evaluate(events: list[dict[str, Any]]) -> dict[str, Any]:
    table = _prec_recall_table(events)
    by = {r["capability"]: r for r in table}
    stop = _stop_mix(events)
    targeted = _targeted_valid_stop_probe(8)

    # Required enrichment fields present?
    required = [
        "action_arguments",
        "recommended_action",
        "query",
        "rendered_context",
        "add_ids",
        "verification_records",
    ]
    field_ok = {k: sum(1 for e in events if k in e) for k in required}

    gates = {
        "Duplicate Precision > 0.9": {
            "pass": (by.get("Duplicate Evidence", {}).get("precision", 0) > 0.9),
            "value": by.get("Duplicate Evidence"),
        },
        "Premature Stop Precision > 0.9": {
            "pass": (by.get("Premature Stop", {}).get("precision", 0) > 0.9),
            "value": by.get("Premature Stop"),
        },
        "Irrelevant Precision > 0.8": {
            "pass": (by.get("Irrelevant Evidence", {}).get("precision", 0) > 0.8),
            "value": by.get("Irrelevant Evidence"),
        },
        "valid stop ENDORSE >= 5 (online or targeted)": {
            "pass": (stop["n_endorse"] >= 5) or targeted["pass"],
            "online_endorse_stops": stop["n_endorse"],
            "targeted": targeted,
        },
        "bad stop CORRECT present": {
            "pass": stop["n_correct"] >= 1,
            "n_correct": stop["n_correct"],
        },
        "0 < P(CORRECT|stop) < 1": {
            "pass": (stop["n_stops"] > 0)
            and (0 < stop["P_CORRECT_given_stop"] < 1),
            "value": stop["P_CORRECT_given_stop"],
            "n_stops": stop["n_stops"],
        },
        "enrichment fields present": {
            "pass": all(field_ok[k] == len(events) for k in required) if events else False,
            "counts": field_ok,
            "n_events": len(events),
        },
    }

    # Training recommendation
    irr_ok = gates["Irrelevant Precision > 0.8"]["pass"]
    dup_ok = gates["Duplicate Precision > 0.9"]["pass"]
    prem_ok = gates["Premature Stop Precision > 0.9"]["pass"]
    stop_ok = (
        gates["valid stop ENDORSE >= 5 (online or targeted)"]["pass"]
        and gates["bad stop CORRECT present"]["pass"]
        and gates["0 < P(CORRECT|stop) < 1"]["pass"]
    )

    train_caps = []
    if dup_ok:
        train_caps.append("Duplicate Evidence")
    if prem_ok and (stop_ok or targeted["pass"]):
        train_caps.append("Premature Stop")
    if irr_ok:
        train_caps.append("Irrelevant Evidence")

    return {
        "n_events": len(events),
        "capability_table": table,
        "stop_mix": stop,
        "targeted_valid_stop": targeted,
        "gates": gates,
        "train_capabilities_recommended": train_caps,
        "exclude_irrelevant": not irr_ok,
        "overall_go_for_dup_premature": dup_ok and prem_ok and (stop_ok or targeted["pass"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.events)
    events = [json.loads(l) for l in path.open() if l.strip()]
    report = evaluate(events)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"n_events={report['n_events']}")
    print("\nCapability table:")
    print(f"{'Capability':28s} Calls Correct Prec Recall")
    for r in report["capability_table"]:
        print(
            f"{r['capability']:28s} {r['calls']:5d} {r['correct']:7d} "
            f"{r['precision']:4.2f} {r['recall']:6.2f}"
        )
    print("\nStop mix:", json.dumps(report["stop_mix"], ensure_ascii=False))
    print("Targeted valid-stop:", json.dumps(report["targeted_valid_stop"], ensure_ascii=False))
    print("\nGates:")
    for k, v in report["gates"].items():
        print(f"  [{'PASS' if v.get('pass') else 'FAIL'}] {k}: {v}")
    print("\ntrain_capabilities_recommended:", report["train_capabilities_recommended"])
    print("exclude_irrelevant:", report["exclude_irrelevant"])
    print("overall_go_for_dup_premature:", report["overall_go_for_dup_premature"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
