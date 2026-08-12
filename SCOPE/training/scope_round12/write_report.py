#!/usr/bin/env python3
"""Write ROUND12_REPORT.md answering the required questions."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs" / "scope_round12"
A = OUT / "phase_a_ckpt_provenance"
B = OUT / "phase_b_operation_boundary"


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    prov = load(A / "SELECTOR_PROVENANCE.json")
    obs = load(A / "CKPT_OBSERVABILITY.json")
    scalar = load(B / "SCALAR_CALIBRATION.json")
    dual = load(B / "DUAL_VIEW_FUSION.json")
    bdec = load(B / "BARRIER_B_DECISION.json")
    root = load(OUT / "ROOT_CAUSE_DECISION.json")
    fl = load(OUT / "FROZEN_LIVE_GATE.json")
    fd = prov.get("first_divergence") or {}
    sels = prov.get("selectors") or {}
    c11 = sels.get("C11L_listwise") or {}
    c9 = sels.get("C9_heuristic_latest") or {}

    smoke_started = (OUT / "phase_d_smoke20").exists() and any((OUT / "phase_d_smoke20").glob("**/*"))
    final_started = (OUT / "phase_e_final100").exists() and any((OUT / "phase_e_final100").glob("**/*"))

    q1 = "NO — not the same evaluation protocol."
    q2 = fd.get("FIRST_DIVERGENCE") or "metric aggregation / missing oracle_op re-pick"
    q3 = "YES" if not obs.get("CKPT_TARGET_NOT_IDENTIFIABLE") else "NO"
    live_s = scalar.get("base_live") or {}
    q4 = (
        "YES"
        if scalar.get("SCALAR_BOUNDARY_REPAIR_PASS")
        else f"NO (live CR={live_s.get('ContinueRecall')} RR={live_s.get('RollbackRecall')} bal={live_s.get('balanced_accuracy')})"
    )
    q5 = (
        "YES — A1 improves CONTINUE recall as an auxiliary boundary signal, but A0 full_stage1 "
        "remains the better primary Stage1 representation for joint CR/RR (Round11 evidence)."
    )
    q6 = (
        f"NO — best learned Stage2 listwise top1={c11.get('top1')} MRR={c11.get('MRR')} "
        f"(gate requires top1>=0.70 / MRR>=0.85)."
    )
    q7 = (
        "NO — Frozen-Live Gate not passed / Phase C mainline not unlocked."
        if not smoke_started
        else "YES — see phase_d_smoke20/"
    )
    q8 = (
        "NO — Smoke20 not passed / not started."
        if not final_started
        else "YES — see phase_e_final100/"
    )
    q9 = "NO / NOT ESTABLISHED"

    lines = []
    lines.append("# ROUND12 REPORT — Checkpoint Provenance → Operation Boundary Repair\n\n")
    lines.append("## Setting\n\n")
    lines.append("- Branch: `scope/round10-rollback-live-parity`\n")
    lines.append("- Outputs: `outputs/scope_round12/`\n")
    lines.append("- Instruction: `H20-0809-todo1.md`\n\n")

    lines.append("## Barrier A — Checkpoint provenance\n\n")
    lines.append("| selector | top1 | MRR |\n|---|---:|---:|\n")
    for name, m in sels.items():
        lines.append(f"| {name} | {m.get('top1')} | {m.get('MRR')} |\n")
    lines.append(f"\n- FIRST_DIVERGENCE: `{q2}`\n")
    lines.append(f"- ROUND9_CKPT_0892_COMPARABLE: `{fd.get('ROUND9_CKPT_0892_COMPARABLE')}`\n")
    lines.append(f"- ROUND9_CKPT_SELECTOR_REFERENCE_VALID: `{fd.get('ROUND9_CKPT_SELECTOR_REFERENCE_VALID')}` ")
    lines.append(f"(canonical C9 top1={c9.get('top1')})\n")
    lines.append(f"- CKPT_TARGET_NOT_IDENTIFIABLE: `{obs.get('CKPT_TARGET_NOT_IDENTIFIABLE')}`\n")
    lines.append(f"- ROOT_CAUSE: {fd.get('ROOT_CAUSE')}\n\n")

    lines.append("## Phase B — Operation boundary\n\n")
    lines.append(f"- tau_scalar: `{scalar.get('tau_scalar')}`\n")
    lines.append(f"- SCALAR_BOUNDARY_REPAIR_PASS: `{scalar.get('SCALAR_BOUNDARY_REPAIR_PASS')}`\n")
    if live_s:
        lines.append(
            f"- base_live @ tau: bal={live_s.get('balanced_accuracy')} "
            f"CR={live_s.get('ContinueRecall')} RR={live_s.get('RollbackRecall')}\n"
        )
    lines.append(f"- DUAL_VIEW_BOUNDARY_REPAIR_PASS: `{dual.get('DUAL_VIEW_BOUNDARY_REPAIR_PASS')}`\n")
    if dual.get("base_live"):
        dl = dual["base_live"]
        lines.append(
            f"- dual-view live: bal={dl.get('balanced_accuracy')} "
            f"CR={dl.get('ContinueRecall')} RR={dl.get('RollbackRecall')} "
            f"(lambda={dual.get('lambda')}, tau={dual.get('tau')})\n"
        )
    lines.append(f"- OPERATION_BOUNDARY_SOLUTION: `{bdec.get('OPERATION_BOUNDARY_SOLUTION')}`\n")
    lines.append(f"- STOP_AFTER_OPERATION_BOUNDARY: `{bdec.get('STOP_AFTER_OPERATION_BOUNDARY')}`\n\n")

    lines.append("## Gates\n\n")
    lines.append(f"- FROZEN_LIVE_GATE.pass: `{fl.get('pass')}` (ran={fl.get('ran')})\n")
    lines.append(f"- ROLLBACK_HARD_CAPABILITY_ESTABLISHED: `{q9}`\n\n")

    lines.append("## Required answers\n\n")
    lines.append(f"1. Round9 0.892 vs Round11 0.627 same protocol? **{q1}**\n")
    lines.append(f"2. First divergence? **{q2}**\n")
    lines.append(f"3. Checkpoint target identifiable under Stage2 effective input? **{q3}**\n")
    lines.append(f"4. Can scalar boundary repair full_stage1 CR gap? **{q4}**\n")
    lines.append(f"5. Should A1 be auxiliary boundary only (not main Stage1)? **{q5}**\n")
    lines.append(f"6. Stage2 top1>=0.70 / MRR>=0.85? **{q6}**\n")
    lines.append(f"7. 20q started? **{q7}**\n")
    lines.append(f"8. 100q started? **{q8}**\n")
    lines.append(f"9. Rollback hard capability established? **{q9}**\n")

    (OUT / "ROUND12_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "ROUND12_REPORT.md")


if __name__ == "__main__":
    main()
