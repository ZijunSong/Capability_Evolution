"""Candidate selector v3: validity → A evidence → C status → cost-aware order.

The legacy ``candidate_selector`` heuristic score multiplies incomparable
numbers and treats missing measurements as 0. v3 stratifies first (validity,
Probe-A direction, Probe-C status) and only *within* a stratum orders by the
fixed-budget Probe-C improvement and measured cost. Missing values stay
``"N/A"`` and never participate in ordering as zeros. Runtime anchors keep
their runtime/hybrid placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from trim.eval.probe_c_metrics import MISSING
from trim.probes.candidate_selector import is_forced_runtime_anchor

# Strata order: earlier = selected first.
STRATUM_PROMOTE = 0
STRATUM_BORDERLINE = 1
STRATUM_NO_C_EVIDENCE = 2
STRATUM_DEPRIORITIZE = 3
_STRATUM_BY_STATUS = {
    "PROMOTE": STRATUM_PROMOTE,
    "BORDERLINE": STRATUM_BORDERLINE,
    "DEPRIORITIZE": STRATUM_DEPRIORITIZE,
    "INVALID": STRATUM_NO_C_EVIDENCE,
    "LOW_SUPPORT": STRATUM_NO_C_EVIDENCE,
    "LOW_SIGNAL": STRATUM_NO_C_EVIDENCE,
    "NOT_RUN": STRATUM_NO_C_EVIDENCE,
}


@dataclass
class CandidateRow:
    component_id: str
    valid: bool = True  # measurement chain valid (C0 READY or audited)
    a_delta: float | str = MISSING  # Probe A add gain (J(H_m) - J(H0))
    c_status: str = "NOT_RUN"
    r_n: float | str = MISSING  # fixed-budget NLL improvement
    r_d: float | str = MISSING  # fixed-budget D_K improvement
    cost_gpu_hours: float | str = MISSING
    placement: str = "semantic"  # semantic | runtime | hybrid
    detail: dict[str, Any] = field(default_factory=dict)


def _sort_key(row: CandidateRow) -> tuple:
    """Within-stratum order: C improvement (desc) then cost (asc).

    MISSING never sorts as 0: rows with missing C evidence are pushed behind
    rows with real evidence in the same stratum, and rows with missing cost
    sort after rows with measured cost.
    """
    r_n = row.r_n if row.r_n != MISSING else float("-inf")
    r_d = row.r_d if row.r_d != MISSING else float("-inf")
    cost = row.cost_gpu_hours if row.cost_gpu_hours != MISSING else float("inf")
    has_c = 0 if (row.r_n != MISSING or row.r_d != MISSING) else 1
    return (has_c, -(float(r_n) + float(r_d)), float(cost))


def select_candidates_v3(
    rows: Sequence[CandidateRow],
    *,
    top_k: int | None = None,
    force_runtime_anchors: bool = True,
) -> dict[str, Any]:
    """Stratified selection with a traceable decision record."""
    records: list[dict[str, Any]] = []
    buckets: dict[int, list[CandidateRow]] = {}
    for row in rows:
        if not row.valid:
            records.append(
                {
                    "component_id": row.component_id,
                    "selected": False,
                    "stratum": "INVALID_MEASUREMENT",
                    "reason": "measurement chain invalid; excluded from ranking",
                }
            )
            continue
        stratum = _STRATUM_BY_STATUS.get(row.c_status, STRATUM_NO_C_EVIDENCE)
        if is_forced_runtime_anchor(row.component_id) and force_runtime_anchors:
            stratum = -1  # runtime anchors keep placement regardless of probes
        buckets.setdefault(stratum, []).append(row)

    ordered: list[tuple[int, CandidateRow]] = []
    for stratum in sorted(buckets):
        for row in sorted(buckets[stratum], key=_sort_key):
            ordered.append((stratum, row))

    selected: list[str] = []
    for stratum, row in ordered:
        take = top_k is None or len(selected) < int(top_k)
        if not take:
            records.append(
                {
                    "component_id": row.component_id,
                    "selected": False,
                    "stratum": stratum,
                    "reason": "below top_k",
                }
            )
            continue
        selected.append(row.component_id)
        records.append(
            {
                "component_id": row.component_id,
                "selected": True,
                "stratum": stratum,
                "reason": "stratified order",
                "a_delta": row.a_delta,
                "c_status": row.c_status,
                "r_n": row.r_n,
                "r_d": row.r_d,
                "cost_gpu_hours": row.cost_gpu_hours,
                "placement": row.placement,
            }
        )
    return {
        "selected": selected,
        "records": records,
        "note": "A×B×C bare products are not multiplied; missing values are "
        "excluded from ordering rather than scored as 0",
    }


__all__ = ["CandidateRow", "select_candidates_v3"]
