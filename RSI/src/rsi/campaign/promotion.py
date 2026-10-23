"""Promotion policy (task book §10.2).

Default: promote when C vs A clears the pre-registered threshold and interval
conditions, no domain/task-family regression exceeds the bound, and protocol +
cost constraints pass.  Recovering ALL Teacher gain is NOT required to
continue — partial internalization may promote with the
``partial_internalization`` flag.  A "full recovery" claim needs a separate
C-vs-B non-inferiority condition; ``surpassed_teacher`` needs separate
positive-difference evidence.  Cost reduction is never assumed: H0 without
external components may still spend more tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsi.evaluation.aggregation import SuiteResult
from rsi.evaluation.retention import FourCells


@dataclass(frozen=True)
class PromotionPolicy:
    min_delta_w: float = 0.0
    require_non_regression: bool = True
    max_domain_regression: float = 0.02
    require_protocol_ok: bool = True
    allow_partial_internalization: bool = True
    require_full_teacher_recovery_for_full_claim: bool = True
    max_cost_increase_ratio: float = 2.0

    @staticmethod
    def load(path: str | Path) -> "PromotionPolicy":
        obj = json.loads(Path(path).read_text("utf-8"))
        cost = dict(obj.get("cost_constraints") or {})
        return PromotionPolicy(
            min_delta_w=float(obj.get("min_delta_w", 0.0)),
            require_non_regression=bool(obj.get("require_non_regression", True)),
            max_domain_regression=float(obj.get("max_domain_regression", 0.02)),
            require_protocol_ok=bool(obj.get("require_protocol_ok", True)),
            allow_partial_internalization=bool(obj.get("allow_partial_internalization", True)),
            require_full_teacher_recovery_for_full_claim=bool(
                obj.get("require_full_teacher_recovery_for_full_claim", True)
            ),
            max_cost_increase_ratio=float(cost.get("max_cost_increase_ratio", 2.0)),
        )


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    flags: Mapping[str, bool]
    four_cells_json: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "flags": dict(self.flags),
            "four_cells": dict(self.four_cells_json),
        }


def evaluate_promotion(
    cells: FourCells,
    policy: PromotionPolicy,
    *,
    protocol_ok: bool = True,
    cost_ratio: float = 1.0,
) -> PromotionDecision:
    flags = cells.flags()
    if policy.require_protocol_ok and not protocol_ok:
        return PromotionDecision(False, "PROTOCOL_OR_COST_CONSTRAINT", flags, cells.to_json())
    if cost_ratio > policy.max_cost_increase_ratio:
        return PromotionDecision(False, f"COST_INCREASE {cost_ratio:.2f} > {policy.max_cost_increase_ratio}", flags, cells.to_json())
    if cells.delta_w < policy.min_delta_w:
        return PromotionDecision(
            False,
            f"MODEL_NOT_PROMOTED: delta_w={cells.delta_w:.4f} < min_delta_w={policy.min_delta_w}",
            flags,
            cells.to_json(),
        )
    if policy.require_non_regression:
        for dom_a, dom_c in zip(cells.a.domains, cells.c.domains):
            if dom_c.score < dom_a.score - policy.max_domain_regression:
                return PromotionDecision(
                    False,
                    f"DOMAIN_REGRESSION {dom_a.domain}: {dom_c.score:.4f} < {dom_a.score:.4f} - {policy.max_domain_regression}",
                    flags,
                    cells.to_json(),
                )
    partial = flags.get("partial_internalization", False)
    if partial and not policy.allow_partial_internalization:
        return PromotionDecision(False, "PARTIAL_INTERNALIZATION_NOT_ALLOWED", flags, cells.to_json())
    label = "partial_internalization" if partial else "full_or_more"
    return PromotionDecision(
        True,
        f"PROMOTED ({label}); retention_ratio={cells.retention_ratio()}",
        flags,
        cells.to_json(),
    )
