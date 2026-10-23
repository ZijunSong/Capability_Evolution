"""Four-cell comparison and retention statistics (task book §10.2).

Cells: A=(theta_k,H0) B=(theta_k,H0+Bk) C=(theta_k+1,H0) D=(theta_k+1,H0+Bk).
``DeltaH = B - A``, ``DeltaW = C - A``,
``retention_ratio = (C - A) / (B - A)`` **only when the denominator is a
reliable positive gain**.  Values are never clipped to look better.  A "full
recovery" claim needs a separate non-inferiority condition of C vs B;
``surpassed_teacher`` needs separate positive-difference evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from rsi.evaluation.aggregation import SuiteResult


@dataclass(frozen=True)
class FourCells:
    a: SuiteResult
    b: SuiteResult
    c: SuiteResult
    d: SuiteResult

    @property
    def delta_h(self) -> float:
        return self.b.macro_score - self.a.macro_score

    @property
    def delta_w(self) -> float:
        return self.c.macro_score - self.a.macro_score

    @property
    def delta_d(self) -> float:
        return self.d.macro_score - self.a.macro_score

    @property
    def c_minus_b(self) -> float:
        return self.c.macro_score - self.b.macro_score

    def retention_ratio(self, *, min_reliable_delta_h: float = 1e-6) -> float | None:
        if self.delta_h <= min_reliable_delta_h:
            return None
        return self.delta_w / self.delta_h

    def flags(self) -> dict[str, bool]:
        ret = self.retention_ratio()
        return {
            "partial_internalization": self.delta_w > 0 and (ret is None or ret < 1.0),
            "full_teacher_recovery": ret is not None and ret >= 1.0 - 1e-9,
            "surpassed_teacher": self.c_minus_b > 0,
            "no_reliable_harness_gain": self.delta_h <= 1e-6,
        }

    def to_json(self) -> dict[str, object]:
        ret = self.retention_ratio()
        return {
            "cells": {
                "A_old_min": self.a.macro_score,
                "B_old_full": self.b.macro_score,
                "C_new_min": self.c.macro_score,
                "D_new_full": self.d.macro_score,
            },
            "delta_h": self.delta_h,
            "delta_w": self.delta_w,
            "delta_d": self.delta_d,
            "c_minus_b": self.c_minus_b,
            "retention_ratio": ret,
            "flags": self.flags(),
        }
