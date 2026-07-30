#!/usr/bin/env python3
"""Round 4 Barrier 1.3: forced KEEP/SKIP episode test on synthetic states."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.dup_decision_point import build_decision_points
from harness.capability.dup_operation import DupOperation
from harness.capability.state import DecisionState
from harness.shadow.action_realizer import ActionRealizer
from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.scope.dup_telemetry import AdmissionEvent, DupTelemetryAggregator
from training.train_rl import CurateTool


def _make_state(
    *,
    curated: tuple[str, ...],
    pool: tuple[str, ...],
    candidate: str,
    query: str,
    episode_id: str,
) -> DecisionState:
    return DecisionState(
        episode_id=episode_id,
        task_id="forced_test",
        turn_id=1,
        query=query,
        rendered_context=f"Query: {query}\nCurated: {list(curated)}\nPool: {list(pool)}\nCandidate: {candidate}",
        action_history=(),
        observation_ids=("obs_0",),
        visible_document_ids=pool,
        pool_document_ids=pool,
        curated_document_ids=curated,
        evidence_claims=(),
        verification_records=(),
        remaining_turns=5,
        remaining_search_calls=None,
        token_budget_used=10,
        token_budget_total=100,
        last_action_type="curate_document",
        repeated_query_score=0.0,
        wm_snapshot_hash="forced",
    )


def build_synthetic_states() -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for i in range(10):
        cid = f"u{i+1}"
        curated = (f"c{i+1}",)
        pool = curated + (cid,)
        states.append(
            {
                "id": f"unique_{i+1:02d}",
                "is_duplicate": False,
                "state": _make_state(
                    curated=curated,
                    pool=pool,
                    candidate=cid,
                    query=f"unique evidence test {i+1}",
                    episode_id=f"unique_{i+1}",
                ),
                "candidate_id": cid,
            }
        )
    for i in range(10):
        cid = f"d{i+1}"
        curated = (cid,)
        pool = curated + (f"x{i+1}",)
        states.append(
            {
                "id": f"duplicate_{i+1:02d}",
                "is_duplicate": True,
                "state": _make_state(
                    curated=curated,
                    pool=pool,
                    candidate=cid,
                    query=f"duplicate evidence test {i+1}",
                    episode_id=f"dup_{i+1}",
                ),
                "candidate_id": cid,
            }
        )
    return states


def run_forced_operation(
    spec: dict[str, Any],
    forced_op: DupOperation,
) -> dict[str, Any]:
    state: DecisionState = spec["state"]
    candidate_id: str = spec["candidate_id"]
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": [candidate_id]},
    )
    shadow = DupBilateralShadow()
    realizer = ActionRealizer()
    points = build_decision_points(state, student)
    shadow_art = shadow.analyze_candidate(state, student, points[0])
    shadow_op = str((shadow_art.metadata or {}).get("shadow_operation", ""))
    route = shadow_art.mode.value.upper()

    cand = realizer.realize_operation(
        state, forced_op, candidate_id=candidate_id, student_action=student
    )
    realized = cand.action if cand else None
    add_ids = (realized.arguments.get("add_ids") or []) if realized else []
    remove_ids = (realized.arguments.get("remove_ids") or []) if realized else []
    curated_before = set(state.curated_document_ids)
    actually_curated = bool(add_ids) and any(str(d) not in curated_before for d in add_ids)

    event = AdmissionEvent(
        candidate_evidence_id=candidate_id,
        candidate_is_duplicate=bool(spec["is_duplicate"]),
        student_operation=forced_op.value,
        shadow_operation=shadow_op,
        route=route,
        realized_runtime_action=realized.to_dict() if realized else None,
        actually_curated=actually_curated,
        query_id=spec["id"],
        turn_id=state.turn_id,
    )
    agg = DupTelemetryAggregator()
    agg.add(event)
    summary = agg.summarize()

    return {
        "state_id": spec["id"],
        "is_duplicate_gt": spec["is_duplicate"],
        "forced_operation": forced_op.value,
        "shadow_operation": shadow_op,
        "shadow_is_duplicate": bool((shadow_art.metadata or {}).get("candidate_is_duplicate")),
        "predicted_operation": forced_op.value,
        "route": route,
        "realized_add_ids": add_ids,
        "realized_remove_ids": remove_ids,
        "actually_curated": actually_curated,
        "telemetry": summary,
        "input": {
            "query": state.query,
            "curated": list(state.curated_document_ids),
            "pool": list(state.pool_document_ids),
            "candidate_id": candidate_id,
        },
    }


def write_report(records: list[dict], out_md: Path) -> None:
    lines = [
        "# Forced Episode Report (Round 4 Barrier 1.3)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Total episodes: {len(records)} (20 states × 2 forced ops)",
        "",
        "## Spot checks (3 unique + 3 duplicate)",
        "",
    ]
    spot_ids = [
        "unique_01",
        "unique_05",
        "unique_10",
        "duplicate_01",
        "duplicate_05",
        "duplicate_10",
    ]
    for rec in records:
        if rec["state_id"] not in spot_ids:
            continue
        lines.extend(
            [
                f"### {rec['state_id']} — forced {rec['forced_operation']}",
                "",
                f"- GT duplicate: {rec['is_duplicate_gt']}",
                f"- Shadow: {rec['shadow_operation']} (dup={rec['shadow_is_duplicate']})",
                f"- Realized add: {rec['realized_add_ids']}, remove: {rec['realized_remove_ids']}",
                f"- Actually curated: {rec['actually_curated']}",
                f"- DCR numerator/denom: {rec['telemetry'].get('duplicate_curate_rate_numerator')}/{rec['telemetry'].get('duplicate_curate_rate_denominator')}",
                f"- FSR numerator/denom: {rec['telemetry'].get('false_skip_rate_numerator')}/{rec['telemetry'].get('false_skip_rate_denominator')}",
                "",
            ]
        )

    # Aggregate sanity
    keep_on_unique = [
        r for r in records if not r["is_duplicate_gt"] and r["forced_operation"] == "KEEP_EVIDENCE"
    ]
    skip_on_dup = [
        r for r in records if r["is_duplicate_gt"] and r["forced_operation"] == "SKIP_DUPLICATE"
    ]
    keep_curated = sum(1 for r in keep_on_unique if r["actually_curated"])
    skip_not_curated = sum(1 for r in skip_on_dup if not r["actually_curated"])

    lines.extend(
        [
            "## Aggregate checks",
            "",
            f"- Unique + KEEP curated: {keep_curated}/{len(keep_on_unique)}",
            f"- Duplicate + SKIP not curated: {skip_not_curated}/{len(skip_on_dup)}",
            "",
            "## DCR / FSR semantics",
            "",
            "- `duplicate_curate_rate` = `n_duplicate_accepted / n_duplicate_gt`",
            "- `false_skip_rate` = `n_unique_rejected / n_unique_gt`",
            "- Rates always include numerator/denominator in telemetry output.",
            "",
        ]
    )
    b1_pass = keep_curated == len(keep_on_unique) and skip_not_curated == len(skip_on_dup)
    lines.append(f"**B1 forced episode PASS:** {b1_pass}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "outputs/scope_round4/metric_audit/forced_episode.jsonl",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_REPO / "outputs/scope_round4/metric_audit/FORCED_EPISODE_REPORT.md",
    )
    args = p.parse_args()

    states = build_synthetic_states()
    records: list[dict] = []
    for spec in states:
        for op in (DupOperation.KEEP_EVIDENCE, DupOperation.SKIP_DUPLICATE):
            records.append(run_forced_operation(spec, op))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_report(records, args.report)
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
