"""Build rollout / policy backends from BiSHOP/.env LLM settings."""

from __future__ import annotations

from typing import Any

from harness.llm_env import get_llm_client, get_llm_model_name, get_llm_settings, llm_api_configured
from training.opd.vllm_rollout_backend import VLLMRolloutBackend


def resolve_policy_backend(
    *,
    policy: str,
    manage_vllm: bool,
    vllm_url: str | None,
) -> str:
    """Return ``api`` or ``vllm`` for harness rollout policy selection."""
    if policy == "api":
        if not llm_api_configured():
            raise ValueError(
                "policy=api requires base_url, api_key, and model_name in BiSHOP/.env"
            )
        return "api"
    if policy == "vllm":
        return "vllm"
    # auto: configured .env LLM API takes precedence over local vLLM
    if llm_api_configured() and vllm_url is None:
        return "api"
    if manage_vllm or vllm_url is not None:
        return "vllm"
    return "vllm"


def build_vllm_rollout_backend_from_env(
    *,
    tokenizer_path: str,
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> VLLMRolloutBackend:
    """Create a chat rollout backend from .env or explicit overrides."""
    settings = get_llm_settings()
    if not settings.is_configured() and base_url is None:
        raise ValueError(
            "LLM API not configured. Set base_url, api_key, and model_name in BiSHOP/.env."
        )
    resolved_base_url = (base_url or settings.base_url).strip().rstrip("/")
    resolved_model = model_name or settings.model_name
    resolved_key = api_key or settings.api_key.get_secret_value()
    return VLLMRolloutBackend(
        base_url=resolved_base_url,
        model_name=resolved_model,
        tokenizer_path=tokenizer_path,
        api_key=resolved_key,
    )


def llm_manifest_fields() -> dict[str, Any]:
    """Metadata fields for rollout manifests when using .env LLM API."""
    settings = get_llm_settings()
    return {
        "llm_backend": "api",
        "base_url": settings.base_url,
        "model_name": settings.model_name,
    }
