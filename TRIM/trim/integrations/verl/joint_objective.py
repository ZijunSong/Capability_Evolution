"""CISPO + projected CE semantics for the verl adapter (T3).

verl loss_mode=cispo uses lower=1-clip_ratio_low, upper=1+clip_ratio_high.
Keep the current [0, 5] interval with clip_ratio_low=1.0, clip_ratio_high=4.0.
"""

from __future__ import annotations

from typing import Any

import torch


def cispo_clip_bounds(*, clip_ratio_low: float, clip_ratio_high: float) -> tuple[float, float]:
    return float(1.0 - clip_ratio_low), float(1.0 + clip_ratio_high)


def verl_cispo_clip_config() -> dict[str, float]:
    return {"clip_ratio_low": 1.0, "clip_ratio_high": 4.0}


def cispo_token_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    *,
    advantage: float | torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_low: float = 0.0,
    clip_high: float = 5.0,
) -> torch.Tensor:
    """Reference CISPO: w = stop_gradient(clamp(exp(Δlogp), 0, 5)).

    Gradient flows through current logprob only. Denominator is masked token count.
    """
    if mask is None:
        mask = torch.ones_like(new_logp)
    new_logp = new_logp.float()
    old_logp = old_logp.float()
    mask = mask.float()
    if not torch.isfinite(new_logp).all() or not torch.isfinite(old_logp).all():
        raise ValueError("non-finite logprobs in CISPO reference")
    ratio = (new_logp - old_logp).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("non-finite CISPO importance ratio")
    weight = ratio.clamp(min=float(clip_low), max=float(clip_high)).detach()
    adv = advantage if torch.is_tensor(advantage) else new_logp.new_tensor(float(advantage))
    denom = mask.sum().clamp_min(1.0)
    return -(weight * adv * new_logp * mask).sum() / denom


def combine_joint_loss(rl_loss: torch.Tensor, opd_loss: torch.Tensor, *, lambda_opd: float) -> torch.Tensor:
    """L_total = L_RL + lambda_opd * L_OPD. lambda is not applied a second time if already baked."""
    return rl_loss + float(lambda_opd) * opd_loss


def assert_one_optimizer_step(n_optimizer_steps: int) -> None:
    if int(n_optimizer_steps) not in {0, 1}:
        raise AssertionError(f"n_optimizer_steps must be 0 or 1, got {n_optimizer_steps}")


def resolved_cispo_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    clip = verl_cispo_clip_config()
    low, high = cispo_clip_bounds(**clip)
    return {
        "loss_mode": "cispo",
        "clip_ratio_low": clip["clip_ratio_low"],
        "clip_ratio_high": clip["clip_ratio_high"],
        "clip_low_threshold": low,
        "clip_high_threshold": high,
        "ppo_epochs": 1,
        "use_kl_loss": False,
        "entropy_coeff": 0.0,
        **cfg,
    }
