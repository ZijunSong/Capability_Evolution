#!/usr/bin/env python3
"""Write ROUND10_FOLLOWUP_REPORT.md + ROOT_CAUSE_DECISION + RUN_MANIFEST + SHA256SUMS."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round10_followup"


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cal = _load(OUT / "phase_a/calibration/CALIBRATION_SUMMARY.json")
    canon = _load(OUT / "CANONICAL_BACKEND_GATE.json")
    phase_b = _load(OUT / "PHASE_B_GATE.json")
    smoke = _load(OUT / "SMOKE20_GATE.json")
    final = _load(OUT / "FINAL100_GATE.json")

    a1_pass = bool(cal.get("GATE_A1_PASS"))
    if a1_pass:
        r10_p8 = "A. 可校准 backend score drift"
    elif canon.get("pass") and canon.get("residual_hf_vllm_is_numerical_only"):
        r10_p8 = "B. unavoidable cross-backend numerical difference"
    else:
        r10_p8 = "C. real decision-contract mismatch"

    main_rows = phase_b.get("main_seed_checks") or []
    answers = phase_b.get("answers") or {}

    stop_phase = None
    if not canon.get("pass"):
        stop_phase = "A"
    elif phase_b and not phase_b.get("pass"):
        stop_phase = "B"
    elif smoke and not smoke.get("pass"):
        stop_phase = "C"
    elif final and not final.get("pass"):
        stop_phase = "D"

    hard = final.get("ROLLBACK_HARD_CAPABILITY") or "NOT ESTABLISHED"
    if not final:
        hard = "NOT ESTABLISHED"

    root = {
        "R10_P8_final": r10_p8,
        "Canonical_single_backend_contract": bool(canon.get("pass")),
        "main_noweight_offline_ContinueRecall": {
            r.get("variant"): r.get("offline_ContinueRecall") for r in main_rows
        },
        "main_noweight_base_live_ContinueRecall": {
            r.get("variant"): r.get("holdout_ContinueRecall") for r in main_rows
        },
        "main_noweight_base_live_balanced_accuracy": {
            r.get("variant"): r.get("holdout_bal_acc") for r in main_rows
        },
        "main_noweight_three_seed_consistent": bool(
            phase_b.get("pass")
            or (
                phase_b.get("seed_span_operation_bal_acc_holdout") is not None
                and float(phase_b["seed_span_operation_bal_acc_holdout"]) <= 0.05
            )
        ),
        "Rollback_hard_capability": hard,
        "STOP_AFTER_PHASE_X": stop_phase,
        f"STOP_AFTER_PHASE_{stop_phase}": True if stop_phase else False,
        "phase_gates": {
            "GATE_A1_PASS": a1_pass,
            "CANONICAL_BACKEND_GATE": bool(canon.get("pass")),
            "PHASE_B_GATE": bool(phase_b.get("pass")) if phase_b else None,
            "SMOKE20_GATE": bool(smoke.get("pass")) if smoke else None,
            "FINAL100_GATE": bool(final.get("pass")) if final else None,
        },
        "phase_b_answers": answers,
        "git_commit": _git(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "ROOT_CAUSE_DECISION.json").write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Round 10 Followup Report",
        "",
        f"- git: `{_git()}`",
        f"- R10-P8 final: **{r10_p8}**",
        f"- Canonical single-backend contract: **{canon.get('pass')}**",
        f"- PHASE_B_GATE: **{phase_b.get('pass')}**",
        f"- SMOKE20_GATE: **{smoke.get('pass')}**",
        f"- FINAL100_GATE: **{final.get('pass')}**",
        f"- Rollback hard capability: **{hard}**",
        f"- STOP_AFTER_PHASE_X: **{stop_phase}**",
        "",
        "## Main noweight seeds (frozen live)",
        "",
    ]
    for r in main_rows:
        lines.append(
            f"- `{r.get('variant')}`: offline CR={r.get('offline_ContinueRecall')} "
            f"live CR={r.get('holdout_ContinueRecall')} live bal={r.get('holdout_bal_acc')} "
            f"pass={r.get('pass')}"
        )
    lines += ["", "## Phase B answers", "", f"```json\n{json.dumps(answers, indent=2)}\n```", ""]
    (OUT / "ROUND10_FOLLOWUP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "branch": "scope/round10-rollback-live-parity",
        "git_commit": _git(),
        "output_root": str(OUT),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gates": root["phase_gates"],
        "decision": root,
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    targets = [
        OUT / "CANONICAL_BACKEND_GATE.json",
        OUT / "PHASE_B_GATE.json",
        OUT / "SMOKE20_GATE.json",
        OUT / "FINAL100_GATE.json",
        OUT / "ROOT_CAUSE_DECISION.json",
        OUT / "RUN_MANIFEST.json",
        OUT / "ROUND10_FOLLOWUP_REPORT.md",
        OUT / "phase_a/calibration/CALIBRATION_SUMMARY.json",
    ]
    lines_sum = []
    for t in targets:
        if t.exists():
            lines_sum.append(f"{_sha256(t)}  {t.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines_sum) + "\n", encoding="utf-8")
    print(json.dumps({"stop_phase": stop_phase, "hard": hard, "r10_p8": r10_p8}, indent=2))


if __name__ == "__main__":
    main()
