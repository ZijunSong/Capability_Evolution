#!/usr/bin/env python3
"""Evaluate a Round13 Stage1 variant on valid/test via vLLM factorized replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round11.score_variant import split_metrics
from training.scope_round9.aggregate_frozen_replay import load_jsonl

DATA = _REPO / "artifacts/datasets/scope_round13/operation_sdi"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--gpu", default="0")
    p.add_argument("--port", type=int, default=18700)
    args = p.parse_args()

    merged = args.variant_dir / "merged"
    if not (merged / "config.json").exists():
        raise SystemExit(f"missing merged model at {merged}")

    inp = DATA / ("valid.jsonl" if args.split == "valid" else "test.jsonl")
    if not inp.exists():
        raise SystemExit(f"missing input {inp}")

    eval_dir = args.variant_dir / f"eval_{args.split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = eval_dir / "canonical_vllm_replay.jsonl"
    os.environ["SCOPE_VLLM_OUT_ROOT"] = str(_REPO / "outputs/scope_round13")
    os.environ["PYTHONPATH"] = str(_REPO)
    subprocess.run(
        [
            sys.executable,
            str(_REPO / "training/scope_round11/run_vllm_factorized_split.py"),
            "--model-path",
            str(merged),
            "--input",
            str(inp),
            "--output",
            str(out_jsonl),
            "--port",
            str(args.port),
            "--gpu",
            str(args.gpu),
        ],
        check=True,
        cwd=_REPO,
    )
    rows = load_jsonl(out_jsonl)
    metrics = split_metrics(rows)
    report = {"split": args.split, "variant_dir": str(args.variant_dir), "metrics": metrics}
    (eval_dir / "METRICS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
