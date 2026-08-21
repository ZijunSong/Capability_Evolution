#!/usr/bin/env python3
"""Formal SR-OPD launcher. Does not import SCOPE training.opd."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.adapters.components import minus_mask
from scape.state.snapshot import capture_snapshot
from scape.training.opd_dataset import (
    ProjectionAudit,
    build_projected_row,
    finalize_audit,
    project_and_materialize,
)
from scape.training.opd_events import HarnessEvent
from scape.training.opd_projection import StudentActionSpaceProjector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Student-Realizable Projected OPD")
    p.add_argument("--opd-mode", choices=("sr_opd", "legacy_same_action"), default="sr_opd")
    p.add_argument("--projection-max-events", type=int, default=8)
    p.add_argument("--projection-max-macro-actions", type=int, default=3)
    p.add_argument("--reject-nonrealizable", action="store_true", default=True)
    p.add_argument("--projection-audit-path", type=Path, required=True)
    p.add_argument("--projection-jsonl", type=Path, required=True)
    p.add_argument("--component-id", required=True)
    p.add_argument("--legacy-teacher-kl-weight", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.opd_mode != "sr_opd":
        raise SystemExit("This launcher is the formal SR-OPD path. Use --opd-mode sr_opd.")
    if args.legacy_teacher_kl_weight != 0.0:
        print(
            f"[sr_opd] legacy_teacher_kl_weight={args.legacy_teacher_kl_weight} is ablation-only",
            flush=True,
        )
    rows = json.loads(Path(args.projection_jsonl).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("rows") or [rows]
    projector = StudentActionSpaceProjector(
        max_anchor_scan_events=args.projection_max_events,
        max_macro_actions=args.projection_max_macro_actions,
    )
    audit = ProjectionAudit()
    emitted: list[dict] = []
    for raw in rows:
        events = [HarnessEvent.from_dict(ev) for ev in raw.get("teacher_events") or []]
        snap_payload = raw.get("student_start_snapshot") or raw.get("snapshot")
        if snap_payload:
            from scape.state.snapshot import EnvironmentSnapshot

            snap = EnvironmentSnapshot.from_dict(snap_payload)
        else:
            snap = capture_snapshot(
                query_id=str(raw.get("query_id") or "q"),
                step=0,
                harness_mask=minus_mask(args.component_id),
                working_memory=dict(raw.get("working_memory") or {}),
            )
        projection, steps = project_and_materialize(
            student_snapshot=snap,
            teacher_events=events,
            student_mask=snap.harness_mask,
            component_id=args.component_id,
            projector=projector,
            audit=audit,
        )
        if projection.kind.value == "reject" and args.reject_nonrealizable:
            emitted.append(
                build_projected_row(
                    query_id=snap.query_id,
                    component_id=args.component_id,
                    student_snapshot=snap,
                    teacher_events=events,
                    projection=projection,
                    projected_steps=[],
                )
            )
            continue
        emitted.append(
            build_projected_row(
                query_id=snap.query_id,
                component_id=args.component_id,
                student_snapshot=snap,
                teacher_events=events,
                projection=projection,
                projected_steps=steps,
            )
        )
    finalize_audit(audit)
    args.projection_audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.projection_audit_path.write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    out_rows = args.projection_audit_path.with_name("projected_rows.jsonl")
    with out_rows.open("w", encoding="utf-8") as handle:
        for row in emitted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"audit": str(args.projection_audit_path), "rows": str(out_rows)}, indent=2))


if __name__ == "__main__":
    main()
