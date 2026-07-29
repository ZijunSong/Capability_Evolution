#!/usr/bin/env python3
"""Closed-loop 100q eval for one merged Round3 variant (8 shards, 1 GPU)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--merged-path", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        shard = f"shard{i}"
        out = args.output_dir / shard
        if (out / "summary.json").exists():
            continue
        subprocess.run(
            [
                sys.executable,
                str(_REPO / "training/scope_round3/hmin_v2_dup_rollout.py"),
                "--output-dir",
                str(out),
                "--manifest",
                str(args.manifest),
                "--shard",
                shard,
                "--n-shards",
                "8",
                "--model-path",
                str(args.merged_path),
                "--vllm-port",
                str(args.port),
                "--dup-operation",
                "--parallel",
                "1",
            ],
            check=True,
            env={
                **dict(__import__("os").environ),
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PYTHONPATH": str(_REPO),
            },
        )

  # merge shards
    from training.scope_round3.wave4_compare import merge_shards

    shards = [args.output_dir / f"shard{i}" for i in range(8)]
    merge_shards(shards, args.output_dir / "merged")
    print(json.dumps(json.loads((args.output_dir / "merged" / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
