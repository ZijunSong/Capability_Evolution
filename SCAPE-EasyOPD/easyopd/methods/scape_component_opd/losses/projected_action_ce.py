from __future__ import annotations

import torch

from .action_ce import masked_action_ce


def projected_action_ce(student_logits: torch.Tensor, target_token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics = masked_action_ce(student_logits, target_token_ids, mask)
    metrics.update({"on_policy_state": True, "target_source": "harness_effect_projection"})
    return loss, metrics
