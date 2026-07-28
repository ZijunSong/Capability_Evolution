"""Tests for BiSHOP/.env LLM configuration."""

from __future__ import annotations

from harness.config import get_config
from harness.llm_env import get_llm_settings


def test_llm_settings_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "base_url=https://api.example.com/v1\n"
        "api_key=sk-test\n"
        "model_name=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    get_llm_settings.cache_clear()
    get_config.cache_clear()

    settings = get_llm_settings()
    assert settings.is_configured()
    assert settings.base_url == "https://api.example.com/v1"
    assert settings.model_name == "test-model"

    client = get_config().get_openai_client()
    assert client.base_url is not None
    assert "api.example.com" in str(client.base_url)

    get_llm_settings.cache_clear()
    get_config.cache_clear()


def test_llm_settings_not_configured_with_placeholders(monkeypatch):
    monkeypatch.setenv("BASE_URL", "")
    monkeypatch.setenv("API_KEY", "EXAMPLE")
    monkeypatch.setenv("MODEL_NAME", "")

    get_llm_settings.cache_clear()
    settings = get_llm_settings()
    assert not settings.is_configured()
    get_llm_settings.cache_clear()
