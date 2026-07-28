"""Deprecated: use HFTrainBackend + VLLMRolloutBackend instead."""

from __future__ import annotations

import warnings

from training.opd.hf_train_backend import HFTrainBackend

warnings.warn(
    "hf_policy_backend.HFPolicyBackend is deprecated; "
    "use hf_train_backend.HFTrainBackend for training and "
    "vllm_rollout_backend.VLLMRolloutBackend for rollout.",
    DeprecationWarning,
    stacklevel=2,
)

HFPolicyBackend = HFTrainBackend

__all__ = ["HFPolicyBackend"]
