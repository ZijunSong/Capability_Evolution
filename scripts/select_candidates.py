#!/usr/bin/env python3
"""Build CAPABILITY_PLACEMENT_MAP from H100 imports or LOCAL_CAL64 provisional rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scape.adapters.components import all_component_ids
from scape.probes.candidate_selector import select_candidates, write_placement_map

REPO = Path(__file__).resolve().parents[1]


def _load_rows(path: Path | None) -> list[dict]:
    if path and path.exists():
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "rows" in data:
                return list(data["rows"])
            if isinstance(data, list):
                return data
            raise ValueError(f"unsupported json shape: {path}")
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    # Provisional placeholders until H100 sync / LOCAL_CAL64 fills real numbers
    rows = []
    for cid in all_component_ids():
        rows.append(
            {
                "component_id": cid,
                "contribution": 0.0,
                "influence_above_null": 0.0,
                "runtime_cost": 1.0,
                "quality_positive": False,
                "provisional": True,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV/JSON contribution×influence table; default = empty provisional",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs" / "scape_prestage",
    )
    ap.add_argument("--top-k", type=int, default=2)
    args = ap.parse_args()

    rows = _load_rows(args.input)
    # coerce numeric fields
    for r in rows:
        for k in ("contribution", "influence_above_null", "runtime_cost", "semantic_fraction"):
            if k in r and r[k] != "" and r[k] is not None:
                r[k] = float(r[k])
        if "quality_positive" in r and not isinstance(r["quality_positive"], bool):
            r["quality_positive"] = str(r["quality_positive"]).lower() in {"1", "true", "yes"}

    result = select_candidates(rows, top_k=args.top_k)
    paths = write_placement_map(result, args.out_dir)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(json.dumps(result["candidates"], indent=2))


if __name__ == "__main__":
    main()
