#!/usr/bin/env python3
"""Build nested micro-overfit datasets D2 ⊂ D8 ⊂ D32 ⊂ D128."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.dup_diagnostics import load_jsonl, write_json
from training.scope_round4.build_overfit128 import sample_balanced_overfit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round3/train.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round5_nested",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_rows = load_jsonl(args.source)
    selected, n_keep, n_skip = sample_balanced_overfit(
        train_rows, n_keep=64, n_skip=64, seed=args.seed,
    )

    sizes = [2, 8, 32, 128]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"seed": args.seed, "subsets": {}}

    for size in sizes:
        n_each = size // 2
        subset = [r for r in selected if _op(r) == "KEEP_EVIDENCE"][:n_each]
        subset += [r for r in selected if _op(r) == "SKIP_DUPLICATE"][:n_each]
        out = args.output_dir / f"train_d{size}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["subsets"][f"D{size}"] = {
            "path": str(out),
            "n_total": len(subset),
            "n_keep": n_each,
            "n_skip": n_each,
        }

    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def _op(sample: dict) -> str:
    from training.scope.compact_target import compact_target_from_sample

    ct = compact_target_from_sample(sample)
    return ct.operation.value if ct else ""


if __name__ == "__main__":
    main()
