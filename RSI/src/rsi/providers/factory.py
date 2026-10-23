"""Policy construction from campaign config."""

from __future__ import annotations

from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.errors import ProtocolViolation


def build_policy(cfg: CampaignConfig, *, provider_override: str | None = None) -> Any:
    policy_cfg = cfg.section("policy")
    provider = provider_override or str(policy_cfg.get("provider", "scripted"))
    if provider == "scripted":
        from rsi.providers.scripted import ScriptedPolicy

        return ScriptedPolicy()
    if provider == "tiny_hf":
        from rsi.providers.tiny_hf import HFPolicy

        return HFPolicy()
    if provider == "hf_path":
        from rsi.providers.tiny_hf import HFPolicy

        model_path = policy_cfg.get("model_path")
        if not model_path or str(model_path).startswith("/path/to") or "INJECT" in str(model_path):
            raise ProtocolViolation(
                "policy.model_path must be a real model path for provider=hf_path"
            )
        resolved = cfg.resolve_path(str(model_path))
        return HFPolicy(model_path=str(resolved))
    if provider == "api":
        raise ProtocolViolation(
            "provider=api is for proposer/assisted mode generation; task rollouts "
            "use a trainable open-weight policy (hf_path) in the main configuration"
        )
    raise ProtocolViolation(f"unknown policy provider {provider!r}")
