#!/usr/bin/env python3
"""Offline-only threshold calibration for a Phase C variant (no base_live labels)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round12.calibrate_boundary import load_jsonl, op_metrics, select_scalar_tau


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-dir", type=Path, required=True)
    args = p.parse_args()
    off_p = args.variant_dir / "eval_offline_valid" / "canonical_vllm_replay.jsonl"
    live_p = args.variant_dir / "eval_holdout" / "canonical_vllm_replay.jsonl"
    if not off_p.exists():
        raise SystemExit(f"missing {off_p}")
    off = load_jsonl(off_p)
    tau, off_m = select_scalar_tau(off)
    out = {"tau_scalar": tau, "offline_valid": off_m}
    if live_p.exists():
        # evaluate once after freeze — allowed as frozen holdout eval
        out["base_live"] = op_metrics(load_jsonl(live_p), threshold=tau)
    (args.variant_dir / "SCALAR_THRESHOLD.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tau": tau, "offline_min": min(off_m["ContinueRecall"], off_m["RollbackRecall"])}, indent=2))


if __name__ == "__main__":
    main()
