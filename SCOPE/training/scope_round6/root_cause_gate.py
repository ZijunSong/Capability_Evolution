#!/usr/bin/env python3
"""Compute ROOT_CAUSE_GATE.json from Phase B artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.common import OUT, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-b-dir", type=Path, default=OUT / "phase_b")
    p.add_argument("--output", type=Path, default=OUT / "phase_b/ROOT_CAUSE_GATE.json")
    args = p.parse_args()
    pb = args.phase_b_dir

    matrix_path = pb / "cross_score_matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {}

    # Runtime parity from parity audits
    adapter_parity = 1.0
    hf_runtime_parity = 1.0
    hash_mismatch = 0
    parity_dir = pb / "parity"
    if parity_dir.exists():
        for f in parity_dir.glob("adapter_merged_*/parity.json"):
            d = json.loads(f.read_text())
            adapter_parity = min(adapter_parity, d.get("prediction_parity", 1.0))
        for f in parity_dir.glob("hf_runtime_*/parity.json"):
            d = json.loads(f.read_text())
            hf_runtime_parity = min(hf_runtime_parity, d.get("prediction_parity", 1.0))
            hash_mismatch += int(d.get("hash_mismatch_count", 0))

    H_RUNTIME = (
        adapter_parity < 0.999
        or hf_runtime_parity < 0.999
        or hash_mismatch > 0
    )

    # Calibration vs shift from matrix
    valid_auroc = {}
    own_auroc = {}
    for tag in ("o7_42", "o7_43", "o7_44"):
        if tag in matrix:
            valid_auroc[tag] = matrix[tag].get("valid522", {}).get("AUROC", 0)
            own_auroc[tag] = matrix[tag].get(tag.replace("o7_", "o7_"), {}).get("AUROC", 0)
            own_key = tag  # o7_42 scores on o7_42 states
            own_auroc[tag] = matrix[tag].get(own_key, {}).get("AUROC", 0)

    H_SHIFT = False
    for tag in ("o7_42", "o7_43", "o7_44"):
        v = valid_auroc.get(tag, 0)
        o = own_auroc.get(tag, 0)
        if v >= 0.98 and (o <= 0.80 or v - o >= 0.15):
            H_SHIFT = True

    H_CALIB = False
    for tag in ("o7_42", "o7_43", "o7_44"):
        if tag not in matrix:
            continue
        own = matrix[tag].get(tag, {})
        zero_bal = own.get("BalancedAcc", 0)
        best5 = own.get("best_DupReject_FSR5", {})
        if own.get("AUROC", 0) >= 0.90 and best5.get("DupRejectRecall", 0) >= 0.20:
            if zero_bal < 0.5:
                H_CALIB = True

    # Feedback from state shift
    H_FEEDBACK = False
    for f in pb.glob("STATE_SHIFT_*.json"):
        d = json.loads(f.read_text())
        if d.get("early_turn_better") and d.get("feature_drift_early_vs_late"):
            H_FEEDBACK = True

    gate = {
        "H_RUNTIME": H_RUNTIME,
        "H_CALIB": H_CALIB and not H_RUNTIME,
        "H_SHIFT": H_SHIFT and not H_RUNTIME,
        "H_FEEDBACK": H_FEEDBACK and not H_RUNTIME,
        "adapter_merged_parity": adapter_parity,
        "hf_runtime_parity": hf_runtime_parity,
        "hash_mismatch_count": hash_mismatch,
        "valid_auroc": valid_auroc,
        "own_auroc": own_auroc,
        "priority_order": ["H_RUNTIME", "H_CALIB", "H_SHIFT", "H_FEEDBACK"],
    }
    write_json(args.output, gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
