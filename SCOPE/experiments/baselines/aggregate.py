"""Aggregate baseline summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def aggregate_baselines(root: Path) -> dict[str, Any]:
    rows = []
    for p in sorted(root.rglob("summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_id": s.get("experiment_id"),
                "status": s.get("status"),
                "n_queries": s.get("n_queries"),
                "path": str(p),
            }
        )
    return {"n": len(rows), "runs": rows}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/iclr_baselines")
    args = p.parse_args()
    print(json.dumps(aggregate_baselines(Path(args.root)), indent=2))
