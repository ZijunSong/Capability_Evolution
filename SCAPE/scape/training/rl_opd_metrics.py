"""Hybrid step logging helpers. Reward and projection coverage stay separate."""

from __future__ import annotations

from typing import Any, Sequence

from scape.training.rl_opd_types import HybridStepMetrics, UPDATE_SKIPPED


def empty_hybrid_metrics(*, policy_version: str, lambda_opd: float) -> HybridStepMetrics:
    return HybridStepMetrics(
        update_type=UPDATE_SKIPPED,
        n_rl_datums=0,
        n_opd_datums=0,
        n_rl_tokens=0,
        n_opd_tokens=0,
        rl_loss_proxy=None,
        opd_nll=None,
        lambda_opd=float(lambda_opd),
        projection_coverage=0.0,
        reject_rate=0.0,
        policy_version=policy_version,
    )


def split_log_groups(metrics: HybridStepMetrics) -> dict[str, dict[str, Any]]:
    """Three log namespaces: rl / opd / hybrid."""
    return {
        "rl": {
            "native_loss_proxy": metrics.rl_loss_proxy,
            "num_datums": metrics.n_rl_datums,
            "num_loss_tokens": metrics.n_rl_tokens,
        },
        "opd": {
            "nll": metrics.opd_nll,
            "num_datums": metrics.n_opd_datums,
            "num_loss_tokens": metrics.n_opd_tokens,
            "projection_coverage": metrics.projection_coverage,
            "reject_rate": metrics.reject_rate,
        },
        "hybrid": {
            "lambda_opd": metrics.lambda_opd,
            "update_type": metrics.update_type,
            "optimizer_steps": metrics.n_optimizer_steps,
            "rl_fb_calls": metrics.n_rl_forward_backward,
            "opd_fb_calls": metrics.n_opd_forward_backward,
            "opd_to_rl_token_ratio": metrics.opd_to_rl_token_ratio,
        },
    }


def mean_reward(rewards: Sequence[float]) -> dict[str, float]:
    vals = [float(x) for x in rewards]
    if not vals:
        return {"reward_mean": 0.0, "reward_std": 0.0, "n": 0.0}
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    return {"reward_mean": mean, "reward_std": var**0.5, "n": float(len(vals))}
