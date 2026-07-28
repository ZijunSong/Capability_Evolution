"""vLLM rollout backend — OpenAI-compatible generation for on-policy trajectories."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

from training.opd._policy_backend import RolloutBackend, RolloutResult


def wait_for_vllm_server(
    base_url: str,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Block until vLLM /v1/models responds."""
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"vLLM server not ready at {base_url} after {timeout_s}s: {last_error}"
    )


class VLLMRolloutBackend(RolloutBackend):
    """Student on-policy rollout via vLLM OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model_name: str = "qwen",
        tokenizer_path: str,
        api_key: str = "EMPTY",
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def rollout_chat(
        self,
        messages: list[dict[str, str]],
        sampling_config: dict[str, Any],
    ) -> RolloutResult:
        max_tokens = int(sampling_config.get("max_new_tokens", 64))
        temperature = float(sampling_config.get("temperature", 0.7))

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
        )
        completion_text = response.choices[0].message.content or ""

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # Re-tokenize continuation aligned with prompt (sampled-token OPD path).
        full_text = prompt_text + completion_text
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        action_ids = full_ids[len(prompt_ids) :]

        return RolloutResult(
            prompt_token_ids=prompt_ids,
            action_token_ids=action_ids,
            text=completion_text,
            metadata={
                "backend": "vllm",
                "model": self.model_name,
                "base_url": self.base_url,
            },
        )
