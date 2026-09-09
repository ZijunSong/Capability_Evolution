"""How checkpoints are exposed as the same chat/tools API (E06)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

KNOWN_BASE_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-4B-Instruct-2507",
    "pat-jj/harness-1",
)

PROTOCOL_CHAT_COMPLETIONS_V1 = "chat_completions_v1"


@dataclass(frozen=True)
class ServedModelIdentity:
    api_base_url: str
    api_model: str
    base_model: str | None = None
    adapter_path: str | None = None
    export_method: str | None = None  # merged | lora | unknown
    revision: str | None = None
    tokenizer: str | None = None
    chat_template: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    quantization: str | None = None
    protocol: str = PROTOCOL_CHAT_COMPLETIONS_V1
    supports_tool_calls: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_tool_calling(self) -> None:
        if not self.supports_tool_calls:
            raise RuntimeError(
                f"served model {self.api_model!r} is not a tool-calling endpoint; "
                "do not treat a plain text completion server as Harness-1 eval"
            )


def identity_from_args(args: Any) -> ServedModelIdentity:
    identity = ServedModelIdentity(
        api_base_url=str(getattr(args, "api_base_url", None) or ""),
        api_model=str(getattr(args, "api_model", None) or getattr(args, "model_name", "") or ""),
        base_model=str(getattr(args, "base_model", None) or getattr(args, "model_name", "") or "") or None,
        adapter_path=str(getattr(args, "adapter", None) or "") or None,
        export_method=str(getattr(args, "adapter_export", None) or "") or None,
        revision=str(getattr(args, "model_revision", None) or "") or None,
        tool_parser=str(getattr(args, "tool_parser", None) or "") or None,
        quantization=str(getattr(args, "quantization", None) or "") or None,
        supports_tool_calls=bool(getattr(args, "supports_tool_calls", True)),
    )
    if not identity.api_base_url:
        raise RuntimeError(
            "Harness-1 official eval requires --api-base-url (OpenAI-compatible "
            "chat/completions with tools). Start a server for openai/gpt-oss-20b, "
            "Qwen/Qwen3-4B-Instruct-2507, pat-jj/harness-1, or a trained checkpoint, "
            "then pass that URL. --run-dir / --adapter only identify weights."
        )
    identity.assert_tool_calling()
    return identity


def vllm_serve_hint(model: str) -> str:
    return (
        "vllm serve {model} --enable-auto-tool-choice "
        "--tool-call-parser openai --served-model-name {model}"
    ).format(model=model)
