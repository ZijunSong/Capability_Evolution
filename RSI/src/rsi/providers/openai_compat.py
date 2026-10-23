"""OpenAI-compatible API client for proposer / assisted-mode generation.

Pure ``urllib`` (no third-party HTTP dependency).  Credentials come from the
environment variable named in the config (``api_key_env``); nothing reads a
legacy ``.env`` implicitly.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from rsi.contracts.errors import InfraError, ProtocolViolation


@dataclass(frozen=True)
class APIMessage:
    role: str
    content: str


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str = "RSI_API_KEY",
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ProtocolViolation(
                f"API key env {self.api_key_env!r} is empty; refuse to read legacy .env files"
            )
        return key

    def chat(self, messages: list[APIMessage], *, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key()}",
            },
            method="POST",
        )
        last: BaseException | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"])
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                continue
        raise InfraError(f"API call failed after retries: {last!r}")

    def identity(self) -> Mapping[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "api_key_env": self.api_key_env}
