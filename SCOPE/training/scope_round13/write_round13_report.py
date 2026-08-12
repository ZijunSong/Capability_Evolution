#!/usr/bin/env python3
"""Assemble ROUND13_REPORT.md + ROOT_CAUSE_DECISION.json + SHA256SUMS."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round13"
TRAIN = OUT / "phase_b_stage1/training"


def load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def mstr(m: dict | None) -> str:
    if not m:
        return "missing"
    return (
        f"bal={float(m.get('balanced_accuracy') or 0):.3f} "
        f"CR={float(m.get('ContinueRecall') or 0):.3f} "
        f"RR={float(m.get('RollbackRecall') or 0):.3f}"
    )


def main() -> None:
    obs = load(OUT / "phase_a_shift/OPERATION_OBSERVABILITY.json") or {}
    shift = load(OUT / "phase_a_shift/DISTRIBUTION_SHIFT.json") or {}
    vgate = load(OUT / "phase_b_stage1/STAGE1_VALID_GATE.json") or {}
    tgate = load(OUT / "phase_b_stage1/STAGE1_TEST_GATE.json") or {}
    s2 = load(OUT / "stage2_audit/STAGE2_DEGENERACY_AUDIT.json") or {}
    s2gate = load(OUT / "stage2_targeted/STAGE2_GATE.json") or {}
    s2data = load(OUT / "stage2_targeted/DATASET_GATE.json") or {}

    stage1_gen = bool(tgate.get("STAGE1_FRESH_GENERALIZATION_PASS"))
    stage2_deg = bool(s2.get("STAGE2_TASK_DEGENERATE"))
    stage2_nt = bool(s2gate.get("STAGE2_NONTRIVIAL_GENERALIZATION_PASS"))
    op_internal = bool(vgate.get("STAGE1_VALID_GATE_PASS")) and stage1_gen
    decision = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "OPERATION_OBSERVABILITY_PASS": obs.get("OPERATION_OBSERVABILITY_PASS"),
        "STAGE1_VALID_GATE_PASS": vgate.get("STAGE1_VALID_GATE_PASS"),
        "STAGE1_FRESH_GENERALIZATION_PASS": stage1_gen,
        "STAGE2_TASK_DEGENERATE": stage2_deg,
        "NONDEGENERATE_STAGE2_DATA_PASS": s2data.get("NONDEGENERATE_STAGE2_DATA_PASS"),
        "STAGE2_VALID_GATE_PASS": s2gate.get("STAGE2_VALID_GATE_PASS"),
        "STAGE2_NONTRIVIAL_GENERALIZATION_PASS": stage2_nt,
        "ROLLBACK_OPERATION_INTERNALIZATION_ESTABLISHED": op_internal,
        "ROLLBACK_CHECKPOINT_SELECTION_INTERNALIZATION_ESTABLISHED": bool(
            s2gate.get("STAGE2_VALID_GATE_PASS") and stage2_nt
        ),
        "ROLLBACK_HARD_CAPABILITY_ESTABLISHED": False,
        "STOP_AFTER_STAGE1_VALID": vgate.get("STOP_AFTER_STAGE1_VALID"),
        "STOP_AFTER_STAGE1_TEST": tgate.get("STOP_AFTER_STAGE1_TEST"),
        "notes": [],
    }
    if not obs.get("OPERATION_OBSERVABILITY_PASS", True):
        decision["notes"].append("Stopped or failed observability gate")
    if vgate.get("STOP_AFTER_STAGE1_VALID"):
        decision["notes"].append(
            "Stopped after Stage1 VALID: no TEST100 / Smoke20 / FINAL100 per STOP RULE"
        )
    if stage2_deg:
        decision["notes"].append(
            "Stage2 task degenerate; canonical resolver remains for checkpoint execution"
        )
    else:
        decision["notes"].append(
            "Stage2 natural task not degenerate; pointer trained for diagnostic comparison"
        )

    (OUT / "ROOT_CAUSE_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")

    per = (vgate.get("per_seed") or {})
    abl = (vgate.get("ablations_valid") or {})
    lines = [
        "# ROUND13_REPORT\n\n",
        f"- created_at: {decision['created_at']}\n",
        f"- git: {subprocess.check_output(['git','rev-parse','HEAD'], cwd=_REPO, text=True).strip()}\n",
        f"- STOP: STOP_AFTER_STAGE1_VALID={vgate.get('STOP_AFTER_STAGE1_VALID')}\n\n",
        "## Stage1 VALID (tau=0, R13_VALID100)\n\n",
    ]
    for v in [
        "r13_onpolicy_querynorm_seed42",
        "r13_onpolicy_querynorm_seed43",
        "r13_onpolicy_querynorm_seed44",
    ]:
        mm = (per.get(v) or {}).get("metrics")
        lines.append(f"- {v}: {mstr(mm)} pass={(per.get(v) or {}).get('pass')}\n")
    lines.append(
        f"- seed_span(bal)={vgate.get('seed_span_balanced_accuracy')} "
        f"STAGE1_VALID_GATE_PASS={vgate.get('STAGE1_VALID_GATE_PASS')}\n"
    )
    for v, mm in abl.items():
        lines.append(f"- ablation {v}: {mstr(mm)}\n")

    lines += [
        "\n## Answers\n\n",
        "1. Round12 threshold repair failed because offline/base_live selected boundaries "
        "do not transfer to fresh on-policy visited-state distributions "
        f"(domain AUC≈{(shift.get('domain_classifier_auc_hist_offline_vs_r13_valid'))}).\n",
        "2. Historical vs fresh shift: see `phase_a_shift/DISTRIBUTION_SHIFT.md` / `.json`.\n",
        f"3. A0 operation-identifiable? "
        f"OPERATION_OBSERVABILITY_PASS={obs.get('OPERATION_OBSERVABILITY_PASS')} "
        f"(conflict_rate={obs.get('conflicting_label_event_rate')}).\n",
        f"4. One-pass on-policy distillation 3-seed VALID pass? "
        f"{vgate.get('STAGE1_VALID_GATE_PASS')} "
        f"(n_pass={vgate.get('n_main_seeds_pass')}, span={vgate.get('seed_span_balanced_accuracy')}). "
        "Seeds are stable but under gate: bal≈0.64 / CR≈0.85 / RR≈0.40–0.44 "
        "(need bal≥0.75 and RR≥0.70).\n",
        "5. Query-norm vs event-uniform: query-norm seed42 RR=0.400 > event-uniform RR=0.344; "
        "bal similar (~0.637). Query-norm slightly better on rollback recall.\n",
        "6. Hard-boundary weight: query-norm hard (RR=0.400) >> nohard (RR=0.283); "
        "hard multiplier contributes materially to RR.\n",
        f"7. Stage2 degenerate? {stage2_deg} "
        f"(H0 latest top1 train≈{(s2.get('train') or {}).get('heuristics', {}).get('H0_latest')}).\n",
        f"8. Natural non-degenerate Stage2 data gate? {s2data.get('NONDEGENERATE_STAGE2_DATA_PASS')} "
        f"(train_n={(s2data.get('train') or {}).get('n')}, "
        f"gold_latest_rate={(s2data.get('train') or {}).get('gold_latest_rate')}).\n",
        f"9. Pointer scorer vs Round11 0.627/0.808: "
        f"STAGE2_VALID_GATE_PASS={s2gate.get('STAGE2_VALID_GATE_PASS')} "
        f"details in `stage2_targeted/STAGE2_GATE.json`.\n",
        f"10. Operation internalization? {decision['ROLLBACK_OPERATION_INTERNALIZATION_ESTABLISHED']}\n",
        f"11. Checkpoint selection internalization? "
        f"{decision['ROLLBACK_CHECKPOINT_SELECTION_INTERNALIZATION_ESTABLISHED']}\n",
        f"12. Full rollback hard capability? {decision['ROLLBACK_HARD_CAPABILITY_ESTABLISHED']}\n",
        "\n## Decision\n\n",
        "Fresh on-policy same-state shadow distillation did **not** clear the Stage1 VALID gate. "
        "Per STOP RULE: no sealed TEST100, no Smoke20, no FINAL100. "
        "Current best scientific claim remains diagnostic: A0 is identifiable, "
        "distribution shift is large, Stage2 natural targets are non-degenerate, "
        "but operation internalization is not established on fresh VALID.\n",
    ]
    (OUT / "ROUND13_REPORT.md").write_text("".join(lines), encoding="utf-8")

    paths = list(OUT.rglob("*.json")) + list(OUT.rglob("*.md"))
    lines = []
    for p in sorted(paths):
        if p.name == "SHA256SUMS":
            continue
        rel = p.relative_to(OUT)
        lines.append(f"{sha256_file(p)}  {rel}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote ROUND13_REPORT.md / ROOT_CAUSE_DECISION.json / SHA256SUMS")


if __name__ == "__main__":
    main()
