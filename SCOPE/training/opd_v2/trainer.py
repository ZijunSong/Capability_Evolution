"""Dual-mode OPD trainer: RL + endorse + correct (does not alter GRPO semantics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from harness.artifacts.schema import GuidanceMode
from training.opd_v2.collator import collate_transitions
from training.opd_v2.correct import compute_correct_loss_batch
from training.opd_v2.dataset import TransitionBuffer
from training.opd_v2.endorse import compute_endorse_loss_batch
from training.opd_v2.transitions import OPDTransitionV2
from training.opd_v2.weighting import ModuleWeightTracker, WeightingConfig


@dataclass
class ScopeOPDConfig:
    lambda_base: float = 0.01
    beta: float = 5.0
    correct_scale: float = 1.0
    label_smoothing: float = 0.0
    endorse_enabled: bool = True
    correct_enabled: bool = True
    weighting: WeightingConfig = field(default_factory=WeightingConfig)


class ScopeOPDTrainer:
    """Combines losses: loss = loss_rl + λ_e * endorse + λ_c * correct."""

    def __init__(
        self,
        config: ScopeOPDConfig | None = None,
        *,
        score_fn: Callable[[str, str], list[float]] | None = None,
    ) -> None:
        self.config = config or ScopeOPDConfig()
        self.score_fn = score_fn
        self.buffer = TransitionBuffer()
        self.weights = ModuleWeightTracker(
            config=self.config.weighting
            if self.config.weighting.enabled
            else WeightingConfig(
                enabled=False,
                lambda_0=self.config.lambda_base,
            )
        )
        # Ensure lambda_0 aligns
        self.weights.config.lambda_0 = self.config.lambda_base

    def add_transitions(self, transitions: list[OPDTransitionV2]) -> None:
        for tr in transitions:
            # Apply per-module weight
            w = self.weights.lambda_for(tr.module_id) / max(self.config.lambda_base, 1e-8)
            # Store absolute module_weight as scale relative to lambda_base
            from dataclasses import replace

            weighted = replace(tr, module_weight=float(w))
            self.buffer.add(weighted)
            self.weights.record_shadow(
                tr.module_id,
                mode=tr.mode.value,
                valid=bool(tr.validity_mask),
                candidate_generated=tr.recommended_action_text is not None,
                candidate_valid=bool(tr.validity_mask and tr.mode == GuidanceMode.CORRECT),
            )

    def compute_opd_losses(self) -> dict[str, float]:
        transitions = self.buffer.all()
        batch = collate_transitions(transitions, score_fn=self.score_fn)
        metrics: dict[str, float] = {}
        endorse_loss = 0.0
        correct_loss = 0.0
        if self.config.endorse_enabled and batch.endorse:
            em = compute_endorse_loss_batch(batch.endorse, beta=self.config.beta)
            metrics.update(em)
            endorse_loss = em["endorse_loss"]
        if self.config.correct_enabled and batch.correct:
            cm = compute_correct_loss_batch(
                batch.correct,
                margin_scale=self.config.correct_scale,
                label_smoothing=self.config.label_smoothing,
            )
            metrics.update(cm)
            correct_loss = cm["correct_loss"]
        metrics["opd_endorse_term"] = self.config.lambda_base * endorse_loss
        metrics["opd_correct_term"] = self.config.lambda_base * correct_loss
        metrics["opd_total"] = metrics["opd_endorse_term"] + metrics["opd_correct_term"]
        return metrics

    def combine_with_rl(self, loss_rl: float, opd_metrics: dict[str, float] | None = None) -> dict[str, float]:
        opd_metrics = opd_metrics or self.compute_opd_losses()
        total = float(loss_rl) + float(opd_metrics.get("opd_total", 0.0))
        out = dict(opd_metrics)
        out["loss_rl"] = float(loss_rl)
        out["loss_total"] = total
        self.weights.step()
        return out

    def train_step_offline(self) -> dict[str, float]:
        """Offline dual-mode step without RL term."""
        return self.combine_with_rl(0.0)

    def export_stats(self) -> dict[str, Any]:
        return {mid: s.to_dict() for mid, s in self.weights.stats.items()}
