"""Project-level verl adapter. Framework owns FSDP2/vLLM; this package owns SEC/OPD contracts."""

from trim.integrations.verl.batch_adapter import expand_episode_rows, training_rows_from_groups
from trim.integrations.verl.joint_objective import (
    assert_one_optimizer_step,
    cispo_clip_bounds,
    cispo_token_loss,
    combine_joint_loss,
    resolved_cispo_config,
    verl_cispo_clip_config,
)
from trim.integrations.verl.trainer_adapter import run_verl_fsdp2_train
from trim.integrations.verl.trajectory import TransitionRecord

__all__ = [
    "TransitionRecord",
    "assert_one_optimizer_step",
    "cispo_clip_bounds",
    "cispo_token_loss",
    "combine_joint_loss",
    "expand_episode_rows",
    "resolved_cispo_config",
    "run_verl_fsdp2_train",
    "training_rows_from_groups",
    "verl_cispo_clip_config",
]
