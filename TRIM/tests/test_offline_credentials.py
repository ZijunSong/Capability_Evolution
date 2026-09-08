from __future__ import annotations

import os

import pytest

from trim.eval.offline_credentials import LOCAL_OPENAI_API_KEY, ensure_local_offline_credentials


def test_ensure_fills_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ensure_local_offline_credentials()
    assert os.environ["OPENAI_API_KEY"] == LOCAL_OPENAI_API_KEY


def test_ensure_replaces_blank_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    ensure_local_offline_credentials()
    assert os.environ["OPENAI_API_KEY"] == LOCAL_OPENAI_API_KEY


def test_ensure_keeps_real_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-user-key")
    ensure_local_offline_credentials()
    assert os.environ["OPENAI_API_KEY"] == "sk-real-user-key"


def test_blank_key_is_enough_for_openai_client_ctor(monkeypatch):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    ensure_local_offline_credentials()
    client = openai.OpenAI()
    assert client.api_key == LOCAL_OPENAI_API_KEY
