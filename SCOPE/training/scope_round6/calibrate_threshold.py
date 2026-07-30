#!/usr/bin/env python3
"""Calibrate margin threshold on shard0 closed-loop states (Round 6 C-CALIB)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.common import MANIFEST, OUT, SEEDS, write_json
from training.scope_round6.metrics import best_dup_reject_at_fsr, direct_behavior_metrics, predict_at_threshold
from training.scope_round6.scorer_utils import load_merged_model, score_samples_hf, scorer_paths_for_tag
from training.scope_round6.state_sources import load_b6_admission_states


def _shard0_qids() -> set[str]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {str(x) for x in data.get("shards", {}).get("shard0", [])}


def calibrate_seed(seed: int, gpu: str) -> dict:
    tag = f"o7_{seed}"
    paths = scorer_paths_for_tag(tag)
    samples = load_b6_admission_states(tag)
    shard0 = _shard0_qids()
    calib = [s for s in samples if str(s.get("query_id")) in shard0]
    model, tokenizer, dev = load_merged_model(paths["merged"], gpu)
    scored = score_samples_hf(model, tokenizer, calib, dev)
    labels = [r.label for r in scored]
    margins = [r.margin for r in scored]
    best = best_dup_reject_at_fsr(labels, margins, 0.05)
    zero = direct_behavior_metrics(labels, predict_at_threshold(margins, 0.0))
    return {
        "seed": seed,
        "n_calib": len(calib),
        "tau": best["threshold"],
        "calibrated": best,
        "threshold_zero": zero,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=OUT / "calibration")
    args = p.parse_args()

    per_seed = {seed: calibrate_seed(seed, args.gpu) for seed in SEEDS}
    taus = [per_seed[s]["tau"] for s in SEEDS]
    tau_shared = sum(taus) / len(taus)

    out = {
        "per_seed": per_seed,
        "tau_shared": tau_shared,
        "tau_seed42": per_seed[42]["tau"],
        "tau_seed43": per_seed[43]["tau"],
        "tau_seed44": per_seed[44]["tau"],
    }
    write_json(args.output_dir / "thresholds.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
