"""Local Learnability: D_pre / D_post / L_m on held-out same-state samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trim.training.tool_opd import learnability_score


@dataclass
class LearnabilityCurve:
    component_id: str
    seed: int
    d_pre: float
    d_post_by_n: dict[int, float]
    invalid_tool_rate_pre: float
    invalid_tool_rate_post_by_n: dict[int, float]

    def L_m(self, n: int, *, eps: float = 1e-8) -> float:
        return learnability_score(self.d_pre, self.d_post_by_n[n], eps=eps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "seed": self.seed,
            "d_pre": self.d_pre,
            "d_post_by_n": {str(k): v for k, v in self.d_post_by_n.items()},
            "L_m_by_n": {str(k): self.L_m(k) for k in self.d_post_by_n},
            "invalid_tool_rate_pre": self.invalid_tool_rate_pre,
            "invalid_tool_rate_post_by_n": {
                str(k): v for k, v in self.invalid_tool_rate_post_by_n.items()
            },
        }


def evaluate_gate_l(
    curves: Sequence[LearnabilityCurve],
    *,
    ns: Sequence[int] = (512, 2000, 8000),
) -> dict[str, Any]:
    """Gate L criteria from H20 training migration plan."""
    if len(curves) < 2:
        return {"pass": False, "reason": "need >=2 seeds", "details": {}}

    details: dict[str, Any] = {}
    directions: list[bool] = []
    scaling_ok = True
    invalid_ok = True
    drop_ok = True

    for curve in curves:
        lm = {n: curve.L_m(n) for n in ns if n in curve.d_post_by_n}
        details[str(curve.seed)] = curve.to_dict()
        # held-out divergence clearly down on largest available n
        avail = [n for n in ns if n in curve.d_post_by_n]
        if not avail:
            return {"pass": False, "reason": "no post metrics", "details": details}
        best_n = max(avail)
        improved = curve.d_post_by_n[best_n] < curve.d_pre - 1e-6
        directions.append(improved)
        drop_ok = drop_ok and improved
        # 2k/8k not systematically worse than 512
        if 512 in curve.d_post_by_n:
            for n in avail:
                if n > 512 and curve.d_post_by_n[n] > curve.d_post_by_n[512] + 1e-3:
                    scaling_ok = False
        # invalid tool rate not up
        for n in avail:
            if curve.invalid_tool_rate_post_by_n.get(n, 0.0) > curve.invalid_tool_rate_pre + 1e-6:
                invalid_ok = False

    seed_agree = all(directions) or (not any(directions) and False)
    # require same direction and positive
    seed_agree = all(directions)

    passed = bool(drop_ok and seed_agree and scaling_ok and invalid_ok)
    reason = "PASS" if passed else "FAIL"
    if not drop_ok:
        reason = "divergence_not_down"
    elif not seed_agree:
        reason = "seed_direction_disagree"
    elif not scaling_ok:
        reason = "scaling_regression"
    elif not invalid_ok:
        reason = "invalid_tool_rate_up"

    return {
        "pass": passed,
        "reason": reason,
        "seed_agree": seed_agree,
        "scaling_ok": scaling_ok,
        "invalid_ok": invalid_ok,
        "details": details,
    }


def aggregate_d(
    per_sample_divergence: Sequence[float],
) -> float:
    if not per_sample_divergence:
        raise ValueError("empty divergences")
    return float(sum(per_sample_divergence) / len(per_sample_divergence))
