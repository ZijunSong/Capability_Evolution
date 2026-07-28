#!/usr/bin/env python3
"""Error-slice evaluation for SCOPE selective internalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ERROR_SLICES = (
    "premature_stop",
    "missing_direct_evidence",
    "invalid_citation",
    "unresolved_conflict",
    "duplicate_search",
    "claim_without_support",
)

REASON_TO_SLICE = {
    "PREMATURE_STOP": "premature_stop",
    "MISSING_DIRECT_EVIDENCE": "missing_direct_evidence",
    "INVALID_CITATION": "invalid_citation",
    "UNRESOLVED_CONFLICT": "unresolved_conflict",
    "REPEATED_QUERY": "duplicate_search",
    "CLAIM_WITHOUT_SUPPORT": "claim_without_support",
}


def _empty_slice() -> dict[str, float]:
    return {"pre_count": 0.0, "post_fixed": 0.0, "fix_rate": 0.0}


def analyze_transitions(
    pre_path: Path | None,
    post_path: Path | None,
) -> dict[str, Any]:
    slices = {s: _empty_slice() for s in ERROR_SLICES}

    def ingest(path: Path | None, *, post: bool) -> None:
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                reason = row.get("reason_code") or row.get("artifact", {}).get("reason_code")
                slice_name = REASON_TO_SLICE.get(str(reason))
                if slice_name is None:
                    # allow explicit slice field
                    slice_name = row.get("error_slice")
                if slice_name not in slices:
                    continue
                if not post:
                    slices[slice_name]["pre_count"] += 1.0
                else:
                    # post: count as fixed if mode was correct→later endorse or success flag
                    fixed = bool(row.get("fixed", row.get("success", False)))
                    if fixed:
                        slices[slice_name]["post_fixed"] += 1.0

    ingest(pre_path, post=False)
    ingest(post_path, post=True)

    for name, stats in slices.items():
        pre = stats["pre_count"]
        if pre > 0:
            stats["fix_rate"] = stats["post_fixed"] / pre
    return {"slices": slices}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pre", type=str, default="", help="Pre-training transitions/errors JSONL")
    p.add_argument("--post", type=str, default="", help="Post-training transitions/errors JSONL")
    p.add_argument("--out", type=str, default="outputs/scope_error_slices/summary.json")
    args = p.parse_args(argv)
    summary = analyze_transitions(
        Path(args.pre) if args.pre else None,
        Path(args.post) if args.post else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
