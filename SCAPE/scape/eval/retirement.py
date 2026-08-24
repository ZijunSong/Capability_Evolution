"""Post-training component retirement gates (SCAPE, not SCOPE ModuleRetirementGate).

Four-grid (single component H_-m or coalition H_-S):
  S0: theta0 + H_full
  S1: theta0 + H_-S
  S2: theta' + H_-S
  S3: theta' + H_full

Strong pass: J(theta', H_-S) > J(theta0, H_full) AND C(H_-S) < C(H_full)
Acceptable:  J non-inferior AND cost materially lower
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def compute_ccr(
    j_s2: float,
    j_s0: float,
    j_s1: float,
    *,
    eps: float = 1e-8,
) -> float:
    """Compensation Capture Ratio: how much of the removal gap is recovered."""
    gap = j_s0 - j_s1
    recovered = j_s2 - j_s1
    if abs(gap) < eps:
        return 1.0 if recovered >= 0 else 0.0
    return float(recovered / gap)


def compute_hrr(j_s3: float, j_s0: float, *, eps: float = 1e-8) -> float:
    """Harness Retention Ratio after training: still benefits from full harness?"""
    return float((j_s3 - j_s0) / (abs(j_s0) + eps))


def evaluate_gate_s(
    grid: Mapping[str, Mapping[str, Any]],
    *,
    quality_key: str = "quality",
    cost_key: str = "cost",
    non_inferior_tol: float = 0.0,
    material_cost_reduction: float = 0.05,
    component_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate Stage S gate from four-grid metrics under H_-S."""
    required = ("S0", "S1", "S2", "S3")
    for k in required:
        if k not in grid:
            raise KeyError(f"missing grid cell {k}")

    j = {k: float(grid[k][quality_key]) for k in required}
    c = {k: float(grid[k][cost_key]) for k in required}

    ccr = compute_ccr(j["S2"], j["S0"], j["S1"])
    hrr = compute_hrr(j["S3"], j["S0"])
    n_m_post = c["S2"]  # runtime cost after retirement under H_-S
    coalition = list(component_ids or grid.get("meta", {}).get("component_ids") or [])
    if not coalition:
        legacy = grid.get("meta", {}).get("component_id")
        if legacy:
            coalition = [str(legacy)]

    strong = j["S2"] > j["S0"] and c["S2"] < c["S0"]
    acceptable = (
        j["S2"] >= j["S0"] - non_inferior_tol
        and (c["S0"] - c["S2"]) >= material_cost_reduction * max(c["S0"], 1e-8)
    )
    # Fail pattern: only teacher agreement improved, closed-loop quality not recovered
    only_local = bool(grid.get("meta", {}).get("local_kl_improved")) and j["S2"] < j["S0"] - non_inferior_tol

    if strong:
        verdict = "STRONG_PASS"
    elif acceptable:
        verdict = "ACCEPTABLE_PASS"
    elif only_local:
        verdict = "FAIL_LOCAL_ONLY"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "pass": verdict in {"STRONG_PASS", "ACCEPTABLE_PASS"},
        "J": j,
        "C": c,
        "CCR_m": ccr,
        "HRR": hrr,
        "N_m_post": n_m_post,
        "component_ids": coalition,
        "harness_condition": f"H_-{{{','.join(coalition)}}}" if coalition else "H_-S",
        "can_claim_retired": verdict in {"STRONG_PASS", "ACCEPTABLE_PASS"},
    }
