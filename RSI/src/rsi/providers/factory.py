"""Policy construction from campaign config."""

from __future__ import annotations

from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.errors import ProtocolViolation


def policy_identity_digest(cfg: CampaignConfig) -> str:
    """PolicyArtifact.policy_id for the configured policy WITHOUT loading weights.

    Mirrors what :class:`rsi.providers.tiny_hf.HFPolicy` records
    (``PolicyArtifact.make(base_model_id, kind="base")``) so multi-worker eval
    can fill its provenance fields without a second 54GB model materialization
    in the parent process (kygt9 OOM chain).
    """
    from rsi.contracts.policy import PolicyArtifact

    policy_cfg = cfg.section("policy")
    provider = str(policy_cfg.get("provider", "scripted"))
    if provider == "tiny_hf":
        base_model_id = "tiny-hf"
    elif provider == "hf_path":
        model_path = policy_cfg.get("model_path")
        if not model_path or str(model_path).startswith("/path/to") or "INJECT" in str(model_path):
            raise ProtocolViolation("policy.model_path must be a real model path for provider=hf_path")
        base_model_id = str(cfg.resolve_path(str(model_path)))
    else:
        raise ProtocolViolation(f"policy_identity_digest: unsupported provider {provider!r}")
    return PolicyArtifact.make(base_model_id, kind="base").policy_id


def build_policy(
    cfg: CampaignConfig,
    *,
    provider_override: str | None = None,
    device: str | None = None,
) -> Any:
    """Build the configured policy.

    ``device`` defaults to cuda when torch sees a GPU (single-node 8-GPU panel
    workers pass ``cuda:0`` under their pinned ``CUDA_VISIBLE_DEVICES``).  On
    cuda the weights load in bfloat16 (rollout-only inference; training
    backends cast at loss level).
    """
    policy_cfg = cfg.section("policy")
    provider = provider_override or str(policy_cfg.get("provider", "scripted"))
    if provider == "scripted":
        from rsi.providers.scripted import ScriptedPolicy

        return ScriptedPolicy()
    if provider == "noop":
        # E0 phase A (audit §4.6/§15): reset then immediately finish — the
        # no-op control that measures the initial-state partial credit.
        from rsi.contracts.action import Finish
        from rsi.providers.scripted import FixedActionPolicy

        return FixedActionPolicy([], base_model_id="e0_noop")
    if provider in ("tiny_hf", "hf_path"):
        from rsi.providers.tiny_hf import HFPolicy, tune_torch_threads

        tune_torch_threads()
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        elif str(device).startswith("cuda"):
            # 显式 cuda 设备（如 RSI_PROPOSER_DEVICE=cuda:7）必须真能用；否则在
            # model.to(device) 里才 RuntimeError，错误信号被 _lazy_init 吞掉难定位。
            try:
                import torch

                if not torch.cuda.is_available():
                    raise ProtocolViolation(
                        f"policy device {device!r} requested but torch.cuda.is_available() is False "
                        f"(torch {torch.__version__}) — node driver vs torch build mismatch; "
                        "fix the torch pin instead of falling back to CPU"
                    )
                idx = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
                if idx >= torch.cuda.device_count():
                    raise ProtocolViolation(
                        f"policy device {device!r} out of range: only "
                        f"{torch.cuda.device_count()} CUDA device(s) visible"
                    )
            except ImportError as e:  # noqa: BLE001
                raise ProtocolViolation(f"policy device {device!r} needs torch: {e}") from e
        dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
        if provider == "tiny_hf":
            return HFPolicy(device=device, dtype=dtype)
        model_path = policy_cfg.get("model_path")
        if not model_path or str(model_path).startswith("/path/to") or "INJECT" in str(model_path):
            raise ProtocolViolation(
                "policy.model_path must be a real model path for provider=hf_path"
            )
        resolved = cfg.resolve_path(str(model_path))
        return HFPolicy(model_path=str(resolved), device=device, dtype=dtype)
    if provider == "api":
        raise ProtocolViolation(
            "provider=api is for proposer/assisted mode generation; task rollouts "
            "use a trainable open-weight policy (hf_path) in the main configuration"
        )
    raise ProtocolViolation(f"unknown policy provider {provider!r}")
