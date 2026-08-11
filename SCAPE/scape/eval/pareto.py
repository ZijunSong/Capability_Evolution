"""Quality-Cost Pareto frontier for runtime recomposition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


COST_DIMENSIONS = (
    "enabled_cognitive_components",
    "rendered_context_tokens",
    "state_serialization_tokens",
    "extra_harness_llm_calls",
    "tool_calls",
    "latency_ms",
    "memory_state_ops",
    "wall_clock_s",
)


def scalar_cost(row: Mapping[str, Any], *, weights: Mapping[str, float] | None = None) -> float:
    w = dict(weights or {})
    total = 0.0
    for dim in COST_DIMENSIONS:
        if dim in row:
            total += float(w.get(dim, 1.0)) * float(row[dim])
    if total == 0.0 and "cost" in row:
        return float(row["cost"])
    return total


def is_dominated(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    quality_key: str = "quality",
    cost_key: str = "cost",
) -> bool:
    """True if a is dominated by b (b has >= quality and <= cost, strict in one)."""
    aq, ac = float(a[quality_key]), float(a[cost_key])
    bq, bc = float(b[quality_key]), float(b[cost_key])
    return (bq >= aq and bc <= ac) and (bq > aq or bc < ac)


def pareto_frontier(
    points: Sequence[Mapping[str, Any]],
    *,
    quality_key: str = "quality",
    cost_key: str = "cost",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, p in enumerate(points):
        point = dict(p)
        point.setdefault(cost_key, scalar_cost(point))
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            qq = dict(q)
            qq.setdefault(cost_key, scalar_cost(qq))
            if is_dominated(point, qq, quality_key=quality_key, cost_key=cost_key):
                dominated = True
                break
        if not dominated:
            out.append(point)
    out.sort(key=lambda r: (-float(r[quality_key]), float(r[cost_key])))
    return out


def main_table(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical four rows for the paper main table."""
    required = {
        "original": "theta0 + H_full",
        "no_train_removal": "theta0 + H_slim",
        "trained_full": "theta' + H_full",
        "scape": "theta' + H_slim",
    }
    table = {}
    for key, label in required.items():
        if key not in rows:
            raise KeyError(f"missing main-table row: {key}")
        item = dict(rows[key])
        item["label"] = label
        item.setdefault("cost", scalar_cost(item))
        table[key] = item
    return table
