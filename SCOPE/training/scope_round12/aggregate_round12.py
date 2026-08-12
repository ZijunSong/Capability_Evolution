#!/usr/bin/env python3
"""Aggregate Round12 gates + ROOT_CAUSE_DECISION + report skeleton."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

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
    fd = prov.get("first_divergence") or {}

    root = {
        "round": 12,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "FIRST_DIVERGENCE": fd.get("FIRST_DIVERGENCE"),
        "ROOT_CAUSE": fd.get("ROOT_CAUSE"),
        "ROUND9_CKPT_0892_COMPARABLE": fd.get("ROUND9_CKPT_0892_COMPARABLE", False),
        "ROUND9_CKPT_SELECTOR_REFERENCE_VALID": fd.get("ROUND9_CKPT_SELECTOR_REFERENCE_VALID"),
        "CKPT_TARGET_NOT_IDENTIFIABLE": obs.get("CKPT_TARGET_NOT_IDENTIFIABLE"),
        "OPERATION_BOUNDARY_SOLUTION": bdec.get("OPERATION_BOUNDARY_SOLUTION"),
        "SCALAR_BOUNDARY_REPAIR_PASS": bdec.get("SCALAR_BOUNDARY_REPAIR_PASS"),
        "DUAL_VIEW_BOUNDARY_REPAIR_PASS": bdec.get("DUAL_VIEW_BOUNDARY_REPAIR_PASS"),
        "STOP_AFTER_OPERATION_BOUNDARY": bdec.get("STOP_AFTER_OPERATION_BOUNDARY"),
        "allow_phase_c_mainline": bdec.get("allow_phase_c_mainline"),
        "selector_summary": {
            k: {"top1": v.get("top1"), "MRR": v.get("MRR")}
            for k, v in (prov.get("selectors") or {}).items()
        },
        "scalar_live": (scalar.get("base_live") or {}),
        "dual_live": (dual.get("base_live") or {}),
    }
    (OUT / "ROOT_CAUSE_DECISION.json").write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")

    # Frozen-live gate only if Phase C completed; otherwise mark not-run
    fl_path = OUT / "FROZEN_LIVE_GATE.json"
    if not fl_path.exists():
        fl = {
            "pass": False,
            "ran": False,
            "reason": (
                "Phase C not unlocked"
                if bdec.get("STOP_AFTER_OPERATION_BOUNDARY")
                else "Phase C pending / not aggregated"
            ),
            "STOP_AFTER_FROZEN_LIVE": True,
        }
        fl_path.write_text(json.dumps(fl, indent=2) + "\n", encoding="utf-8")

    # Write / refresh report
    subprocess.run([sys.executable, str(_REPO / "training/scope_round12/write_report.py")], check=False)

    # SHA256SUMS for key artifacts
    paths = [
        A / "SELECTOR_PROVENANCE.json",
        A / "CKPT_METRIC_PARITY.md",
        A / "CKPT_OBSERVABILITY.json",
        B / "CROSS_VIEW_MATRIX.json",
        B / "SCALAR_CALIBRATION.json",
        B / "DUAL_VIEW_FUSION.json",
        B / "BARRIER_B_DECISION.json",
        OUT / "ROOT_CAUSE_DECISION.json",
        OUT / "FROZEN_LIVE_GATE.json",
        OUT / "ROUND12_REPORT.md",
    ]
    lines = []
    for p in paths:
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(root, indent=2))


if __name__ == "__main__":
    main()
