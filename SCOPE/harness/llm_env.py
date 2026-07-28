"""Minimal LLM settings for OpenAI-compatible APIs (e.g. MIFY / Kimi)."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from harness.config import DEFAULT_ENV_FILES


class LlmSettings(BaseSettings):
    """Load ``base_url``, ``api_key``, and ``model_name`` from BiSHOP/.env."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = ""
    api_key: SecretStr = SecretStr("")
    model_name: str = ""

    def is_configured(self) -> bool:
        key = self.api_key.get_secret_value().strip()
        return bool(
            self.base_url.strip()
            and key
            and key.upper() != "EXAMPLE"
            and self.model_name.strip()
        )

    def get_client(self) -> OpenAI:
        if not self.is_configured():
            raise ValueError(
                "LLM API not configured. Set base_url, api_key, and model_name "
                "in BiSHOP/.env (OpenAI-compatible endpoint)."
            )
        return OpenAI(
            base_url=self.base_url.strip().rstrip("/"),
            api_key=self.api_key.get_secret_value(),
        )


@lru_cache(maxsize=1)
def get_llm_settings() -> LlmSettings:
    return LlmSettings()  # type: ignore[call-arg]


def llm_api_configured() -> bool:
    return get_llm_settings().is_configured()


def get_llm_client() -> OpenAI:
    return get_llm_settings().get_client()


def get_llm_model_name() -> str:
    settings = get_llm_settings()
    if not settings.is_configured():
        raise ValueError(
            "LLM API not configured. Set base_url, api_key, and model_name in BiSHOP/.env."
        )
    return settings.model_name.strip()
