"""Explicit threshold / bias for KEEP vs SKIP admission decisions (Round 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.capability.dup_operation import DupOperation
from training.scope.decide_dup_operation import decide_dup_operation


@dataclass(frozen=True)
class DupDecisionConfig:
    """margin = score_skip - score_keep; SKIP iff margin >= threshold + decision_bias."""

    threshold: float = 0.0
    decision_bias: float = 0.0

    def effective_threshold(self) -> float:
        return self.threshold + self.decision_bias

    def predict_from_scores(self, score_keep: float, score_skip: float) -> DupOperation:
        return decide_dup_operation(
            score_keep=score_keep,
            score_skip=score_skip,
            threshold=self.effective_threshold(),
        ).predicted_operation

    def predict_from_margin(self, margin: float) -> DupOperation:
        if margin >= self.effective_threshold():
            return DupOperation.SKIP_DUPLICATE
        return DupOperation.KEEP_EVIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "decision_bias": self.decision_bias,
            "effective_threshold": self.effective_threshold(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DupDecisionConfig:
        return cls(
            threshold=float(d.get("threshold", 0.0)),
            decision_bias=float(d.get("decision_bias", 0.0)),
        )

    @classmethod
    def load(cls, path: Path) -> DupDecisionConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


DEFAULT_DECISION_CONFIG = DupDecisionConfig()
