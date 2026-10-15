"""Four-grid promotion on a frozen checkpoint. S3-only gains do not publish a model."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trim.rsi.statistics import INSUFFICIENT, cluster_paired_delta

PARTIAL = "PROMOTED_PARTIAL_INTERNALIZATION"
FULL = "PROMOTED_FULL_RETIREMENT"
REJECT = "STOP_WEIGHT_PROMOTION_FAILED"
NA = "NA"


def compensation_ratio(j_s2: float, j_s0: float, j_s1: float) -> float | str:
    gap = j_s0 - j_s1
    if gap <= 1e-8:
        return NA
    return (j_s2 - j_s1) / gap


def evaluate_promotion(
    *,
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    delta_weight: float = 0.01,
    lcb_floor: float = 0.0,
    retention_margin: float = 0.02,
    retirement_margin: float = 0.01,
    require_cost_reduction: bool = True,
    allow_partial: bool = True,
    n_boot: int = 400,
    seed: int = 20261015,
) -> dict[str, Any]:
    """``cells`` maps S0..S3 to row lists with cluster_id, query_id, quality, cost.

    Retention rows, when present, live under ``cells['retention']`` with
    ``j_base`` = old-capability quality of theta_r and ``j_other`` of theta_r+1.
    """
    param_rows = _paired(cells["S1"], cells["S2"])
    param = cluster_paired_delta(param_rows, n_boot=n_boot, seed=seed)
    means = {name: _mean_quality(cells[name]) for name in ("S0", "S1", "S2", "S3")}
    costs = {name: _mean_cost(cells[name]) for name in ("S0", "S1", "S2", "S3")}
    ccr = compensation_ratio(means["S2"], means["S0"], means["S1"])
    retention_rows = list(cells.get("retention") or [])
    if retention_rows:
        retention = cluster_paired_delta(retention_rows, n_boot=n_boot, seed=seed + 1)
        retention_ok = (
            retention["status"] == "ok"
            and retention["mean_delta"] is not None
            and float(retention["mean_delta"]) >= -retention_margin
        )
    else:
        retention = {"status": INSUFFICIENT, "mean_delta": None, "lcb": None, "interval": None}
        retention_ok = False
    param_ok = (
        param["status"] == "ok"
        and param["mean_delta"] is not None
        and float(param["mean_delta"]) >= delta_weight
        and param["lcb"] is not None
        and float(param["lcb"]) > lcb_floor
    )
    s2_only = param_ok and retention_ok
    full = False
    if s2_only and ccr != NA:
        noninferior = means["S2"] >= means["S0"] - retirement_margin
        cost_down = costs["S2"] < costs["S0"]
        full = bool(noninferior and (cost_down or not require_cost_reduction))
    if s2_only and full:
        status = FULL
        promotable = True
    elif s2_only and allow_partial:
        status = PARTIAL
        promotable = True
    else:
        status = REJECT
        promotable = False
    return {
        "status": status,
        "promotable": promotable,
        "means": means,
        "costs": costs,
        "delta_param": param,
        "ccr": ccr,
        "retention": retention,
        "retention_ok": retention_ok,
        "rollback_to_parent": not promotable,
    }


def _paired(base_rows: Sequence[Mapping[str, Any]], other_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    other = {(str(row["cluster_id"]), str(row["query_id"]), row.get("seed")): row for row in other_rows}
    paired: list[dict[str, Any]] = []
    for row in base_rows:
        key = (str(row["cluster_id"]), str(row["query_id"]), row.get("seed"))
        match = other.get(key)
        paired.append(
            {
                "cluster_id": row["cluster_id"],
                "query_id": row["query_id"],
                "seed": row.get("seed"),
                "j_base": row.get("quality"),
                "j_other": None if match is None else match.get("quality"),
            }
        )
    return paired


def _mean_quality(rows: Sequence[Mapping[str, Any]]) -> float:
    values = [float(row["quality"]) for row in rows]
    if not values:
        raise ValueError("four-grid cell is empty")
    return sum(values) / len(values)


def _mean_cost(rows: Sequence[Mapping[str, Any]]) -> float:
    values = [float(row.get("cost") or 0.0) for row in rows]
    if not values:
        raise ValueError("four-grid cell is empty")
    return sum(values) / len(values)
