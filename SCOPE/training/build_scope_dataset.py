#!/usr/bin/env python3
"""Build SCOPE v3 supervision dataset from online DecisionState audit events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.scope.dataset_builder import (
    DatasetBuildConfig,
    build_dataset_from_events,
    load_events_jsonl,
    write_split_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True, help="Audit events JSONL")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--valid-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-ignore", action="store_true")
    ap.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help="Optional JSON file merged into dataset manifest.provenance",
    )
    args = ap.parse_args()

    events = load_events_jsonl(args.events)
    provenance = {}
    if args.provenance_json and args.provenance_json.exists():
        provenance = json.loads(args.provenance_json.read_text())
    provenance.update(
        {
            "events_path": str(args.events),
            "n_events_in": len(events),
        }
    )

    cfg = DatasetBuildConfig(
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        drop_ignore=bool(args.drop_ignore),
    )
    train, valid, manifest = build_dataset_from_events(
        events, cfg, provenance=provenance
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_split_jsonl(out / "train.jsonl", train)
    write_split_jsonl(out / "valid.jsonl", valid)
    (out / "stats.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "scope.dataset.v3",
                "train": "train.jsonl",
                "valid": "valid.jsonl",
                "stats": "stats.json",
                **manifest.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(manifest.to_dict(), indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
