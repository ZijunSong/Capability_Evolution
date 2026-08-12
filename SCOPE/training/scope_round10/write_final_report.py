#!/usr/bin/env python3
"""Write ROUND10_FINAL_REPORT.md from artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import OUT, DATA


def _load(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "ROUND10_FINAL_REPORT.md")
    args = p.parse_args()

    replan = _load(DATA / "schema_audit/REPLAN_GATE.json")
    calib = _load(OUT / "calibration/BINARY_CALIBRATION.json")
    gate = _load(OUT / "ROUND10_OFFLINE_GATE.json")
    prior = _load(OUT / "prior_shift/per_model_metrics.json")

    lines = [
        "# Round 10 Final Report",
        "",
        "## 1. REPLAN support?",
        f"- Route {replan.get('route', '?')}: REPLAN_SUPPORTED={replan.get('ROUND10_REPLAN_SUPPORTED')}",
        f"- train/valid/test REPLAN: {replan.get('genuine_replan_train')}/{replan.get('genuine_replan_valid')}/{replan.get('genuine_replan_test')}",
        "",
        "## 2. Binary vs three-class?",
        f"- Primary task: {'three-class' if replan.get('ROUND10_REPLAN_SUPPORTED') else 'binary CONTINUE vs ROLLBACK_TO'}",
        "",
        "## 3. Prior shift contribution",
        "- See outputs/scope_round10/prior_shift/PRIOR_SHIFT_REPORT.md",
        f"- O7 seed42 live_test ContinueRecall (uncalibrated): "
        f"{(prior.get('rollback_o7_seed42') or {}).get('live_test', {}).get('ContinueRecall', 'n/a')}",
        "",
        "## 4. Shared threshold sufficient?",
        f"- Calibration pass: {calib.get('calibration_pass')}",
        f"- tau_shared: {calib.get('tau_shared')}",
        f"- ROUND10_PRIMARY_CAUSE: {calib.get('ROUND10_PRIMARY_CAUSE')}",
        "",
        "## 5. Support-aligned data fixes live transfer?",
        f"- Offline gate pass: {gate.get('offline_gate_pass')}",
        "",
        "## 6. Checkpoint selector",
        "- Frozen operation_only training; Stage 2 not retrained (Round 9 oracle ckpt ~0.89).",
        "",
        "## 7. Enter 20q/100q?",
        f"- Offline gate: {gate.get('offline_gate_pass')} → "
        f"{'smoke20 eligible' if gate.get('offline_gate_pass') else 'blocked'}",
        "",
        "## 8. Stop Rule?",
    ]
    calib_fail = not calib.get("calibration_pass", False)
    train_fail = not gate.get("offline_gate_pass", False)
    if calib_fail and train_fail:
        lines.append("- **TRIGGERED**: calibration and support-aligned training both failed live_test >= 0.70")
    else:
        lines.append("- Not triggered.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
