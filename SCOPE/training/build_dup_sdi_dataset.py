#!/usr/bin/env python3
"""Build Dup-only SDI dataset (query-level split) from v3 supervision samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.scope.dup_dataset_builder import build_dup_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples",
        type=Path,
        default=_REPO_ROOT
        / "outputs/scope_v3_audit_100q/natural_100q/samples.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts/datasets/dup_sdi_round1",
    )
    p.add_argument("--valid-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train, valid, manifest = build_dup_dataset(
        args.samples,
        args.out_dir,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        provenance={
            "source": str(args.samples),
            "capability": "duplicate_evidence",
            "train_mask": 1,
            "split": "query_level",
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "manifest": manifest.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
