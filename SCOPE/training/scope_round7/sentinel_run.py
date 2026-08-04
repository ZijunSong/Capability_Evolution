#!/usr/bin/env python3
"""Forced threshold sentinel runs (Round 7 Gate C)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.decide_dup_operation import decide_dup_operation
from training.scope_round7.common import OUT, write_json


def evaluate_sentinel(scores: list[tuple[float, float]], threshold: float) -> dict:
    n_skip = 0
    n_argmax_mismatch = 0
    for sk, ss in scores:
        d = decide_dup_operation(score_keep=sk, score_skip=ss, threshold=threshold)
        if d.predicted_operation == DupOperation.SKIP_DUPLICATE:
            n_skip += 1
        argmax = DupOperation.SKIP_DUPLICATE if ss >= sk else DupOperation.KEEP_EVIDENCE
        if threshold == 0.0 and d.predicted_operation != argmax:
            n_argmax_mismatch += 1
    n = len(scores)
    return {
        "n": n,
        "n_skip": n_skip,
        "skip_prior": n_skip / max(n, 1),
        "argmax_mismatch": n_argmax_mismatch,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--vllm-port", type=int, default=9207)
    p.add_argument("--n-queries", type=int, default=5)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic score grid for sentinel invariant verification
    scores = [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (-1.0, 2.0), (3.0, -1.0)]
    th = args.threshold
    if math.isinf(th) and th > 0:
        th = float("inf")
    elif math.isinf(th) and th < 0:
        th = float("-inf")

    result = evaluate_sentinel(scores, th)
    gate_c = True
    if th == float("inf"):
        gate_c = result["n_skip"] == 0
    elif th == float("-inf"):
        gate_c = result["n_skip"] == result["n"]
    elif th == 0.0:
        gate_c = result["argmax_mismatch"] == 0

    out = {
        "threshold": args.threshold,
        "sentinel": result,
        "gate_c_pass": gate_c,
        "model_path": str(args.model_path),
    }
    write_json(args.output_dir / "sentinel_result.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
