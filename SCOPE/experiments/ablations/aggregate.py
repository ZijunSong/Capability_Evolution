"""Aggregate ablation summaries across seeds / variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.scope.paired_stats import seed_mean_std


def load_summaries(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def aggregate_group(root: Path, *, metric_path: str = "metrics.balanced_accuracy") -> dict[str, Any]:
    summaries = load_summaries(root)
    by_variant: dict[str, list[float]] = {}
    for s in summaries:
        eid = s.get("experiment_id", "unknown")
        variant = eid
        cur: Any = s
        for part in metric_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is None:
            continue
        by_variant.setdefault(variant, []).append(float(cur))
    out = {v: seed_mean_std(vs) for v, vs in by_variant.items()}
    return {"n_summaries": len(summaries), "by_variant": out, "metric_path": metric_path}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--metric-path", default="metrics.balanced_accuracy")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    report = aggregate_group(Path(args.root), metric_path=args.metric_path)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
