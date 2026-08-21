from __future__ import annotations

from .losses import alpha_jsd, forward_kl_exact, masked_action_ce, projected_action_ce, reverse_kl_exact

LOSS_REGISTRY = {
    "forward_kl": forward_kl_exact,
    "reverse_kl": reverse_kl_exact,
    "jsd": alpha_jsd,
    "action_ce": masked_action_ce,
    "projected_action_ce": projected_action_ce,
}
