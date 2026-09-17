"""Central model-family profiles for eval, vLLM serve, and training rollouts.

Maps checkpoint names / paths to:
- prompt stack (Harmony o200k vs Hugging Face chat + tools)
- vLLM ``--tool-call-parser`` / ``--reasoning-parser`` flags
- smoke-test and retry-prompt routing
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

FAMILY_GPTOSS = "gpt-oss"
FAMILY_QWEN3 = "qwen3"
FAMILY_HF_CHAT = "hf_chat"

STACK_HARMONY = "harmony"
STACK_HF_CHAT = "hf_chat_tools"

DEFAULT_MAX_MODEL_LEN = 32768

_HARMONY_NAME_MARKERS = (
    "gpt-oss",
    "gpt_oss",
    "harness-1",
    "harness1",
    "openai/gpt-oss",
)

_QWEN35_RE = re.compile(r"qwen3[\._-]?5", re.I)
_QWEN_RE = re.compile(r"qwen", re.I)
_GLM47_RE = re.compile(r"glm[\._-]?4[\._-]?7|glm4\.7|glm-4-7", re.I)
_GLM_RE = re.compile(r"glm|chatglm", re.I)
_GEMMA3_RE = re.compile(r"gemma[\._-]?3|gemma3", re.I)
_GEMMA_RE = re.compile(r"gemma", re.I)


@dataclass(frozen=True)
class ModelProfile:
    family: str
    stack: str
    tool_call_parser: str
    reasoning_parser: str | None = None
    moe_backend: str | None = None
    needs_tool_smoke_test: bool = False
    strict_qwen_prompt_ids: bool = False
    extra_vllm_args: tuple[str, ...] = ()
    label: str = ""

    @property
    def is_harmony(self) -> bool:
        return self.stack == STACK_HARMONY

    @property
    def is_hf_chat(self) -> bool:
        return self.stack == STACK_HF_CHAT


HARMONY_PROFILE = ModelProfile(
    family=FAMILY_GPTOSS,
    stack=STACK_HARMONY,
    tool_call_parser="openai",
    moe_backend="triton",
    needs_tool_smoke_test=False,
    label="gpt-oss / harness-1",
)

QWEN3_PROFILE = ModelProfile(
    family=FAMILY_QWEN3,
    stack=STACK_HF_CHAT,
    tool_call_parser="hermes",
    needs_tool_smoke_test=True,
    strict_qwen_prompt_ids=True,
    label="Qwen3",
)

QWEN35_PROFILE = ModelProfile(
    family=FAMILY_HF_CHAT,
    stack=STACK_HF_CHAT,
    tool_call_parser="qwen3_coder",
    reasoning_parser="qwen3",
    needs_tool_smoke_test=True,
    label="Qwen3.5",
)

GLM47_PROFILE = ModelProfile(
    family=FAMILY_HF_CHAT,
    stack=STACK_HF_CHAT,
    tool_call_parser="glm47",
    needs_tool_smoke_test=True,
    label="GLM-4.7",
)

GLM45_PROFILE = ModelProfile(
    family=FAMILY_HF_CHAT,
    stack=STACK_HF_CHAT,
    tool_call_parser="glm45",
    needs_tool_smoke_test=True,
    label="GLM-4",
)

_GEMMA3_PYTHONIC_CHAT_TEMPLATE = str(
    Path(__file__).resolve().parent / "templates" / "tool_chat_template_gemma3_pythonic.jinja"
)

GEMMA_PROFILE = ModelProfile(
    family=FAMILY_HF_CHAT,
    stack=STACK_HF_CHAT,
    tool_call_parser="pythonic",
    needs_tool_smoke_test=True,
    extra_vllm_args=(
        "--language-model-only",
        "--generation-config",
        "vllm",
        "--chat-template",
        _GEMMA3_PYTHONIC_CHAT_TEMPLATE,
    ),
    label="Gemma",
)

GENERIC_HF_CHAT_PROFILE = ModelProfile(
    family=FAMILY_HF_CHAT,
    stack=STACK_HF_CHAT,
    tool_call_parser="hermes",
    needs_tool_smoke_test=True,
    label="hf_chat (generic)",
)


def _norm_name(source: str) -> str:
    return str(source or "").strip().lower()


def _name_matches(name: str, markers: Sequence[str]) -> bool:
    n = _norm_name(name)
    return any(m in n for m in markers)


def classify_profile_by_name(source: str) -> ModelProfile | None:
    """Return a profile from model id / path substrings, or None if unknown."""
    n = _norm_name(source)
    if not n:
        return None
    if _name_matches(n, _HARMONY_NAME_MARKERS):
        return HARMONY_PROFILE
    if _QWEN35_RE.search(n):
        return QWEN35_PROFILE
    if _QWEN_RE.search(n):
        return QWEN3_PROFILE
    if _GLM47_RE.search(n):
        return GLM47_PROFILE
    if _GLM_RE.search(n):
        return GLM45_PROFILE
    if _GEMMA3_RE.search(n) or _GEMMA_RE.search(n):
        return GEMMA_PROFILE
    return None


def _tokenizer_harmony_marker(tokenizer: Any) -> bool:
    try:
        call = tokenizer.convert_tokens_to_ids("<|call|>")
    except Exception:
        return False
    return call == 200012


def _tokenizer_hf_chat_marker(tokenizer: Any) -> bool:
    for tok in ("<|im_end|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid not in {None, -1}:
            return True
    return False


def resolve_model_profile(source: str, tokenizer: Any | None = None) -> ModelProfile:
    """Resolve eval / serve profile from checkpoint name and optional tokenizer."""
    by_name = classify_profile_by_name(source)
    if by_name is not None:
        return by_name
    if tokenizer is not None:
        if _tokenizer_harmony_marker(tokenizer):
            return HARMONY_PROFILE
        if _tokenizer_hf_chat_marker(tokenizer):
            if _QWEN_RE.search(_norm_name(source)):
                return QWEN3_PROFILE
            return GENERIC_HF_CHAT_PROFILE
    return HARMONY_PROFILE


def is_harmony_model(model: str | None, tokenizer: Any | None = None) -> bool:
    return resolve_model_profile(str(model or ""), tokenizer).is_harmony


def is_hf_chat_family(family: str | None) -> bool:
    name = str(family or "").lower()
    return name in {FAMILY_QWEN3, FAMILY_HF_CHAT, "qwen", "qwen3_chat", "hf_chat_tools"}


def vllm_extra_args(
    source: str,
    *,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tokenizer: Any | None = None,
) -> list[str]:
    profile = resolve_model_profile(source, tokenizer)
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        profile.tool_call_parser,
        "--max-model-len",
        str(int(max_model_len)),
        "--trust-remote-code",
    ]
    if profile.reasoning_parser:
        args.extend(["--reasoning-parser", profile.reasoning_parser])
    if profile.moe_backend:
        args.extend(["--moe-backend", profile.moe_backend])
    if profile.extra_vllm_args:
        args.extend(profile.extra_vllm_args)
    return args


def vllm_extra_shell(source: str, *, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> str:
    return " ".join(vllm_extra_args(source, max_model_len=max_model_len))


def needs_tool_call_smoke_test(source: str, tokenizer: Any | None = None) -> bool:
    return resolve_model_profile(source, tokenizer).needs_tool_smoke_test


def tool_call_parser_for(source: str, tokenizer: Any | None = None) -> str:
    return resolve_model_profile(source, tokenizer).tool_call_parser


KNOWN_BASE_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3.5-35B-A3B",
    "THUDM/glm-4-9b-chat",
    "THUDM/GLM-4.6V",
    "google/gemma-3-4b-it",
    "pat-jj/harness-1",
)


def _cli_vllm_extra(model: str) -> int:
    sys.stdout.write(vllm_extra_shell(model) + "\n")
    return 0


def _cli_tool_parser(model: str) -> int:
    sys.stdout.write(tool_call_parser_for(model) + "\n")
    return 0


def _cli_needs_smoke_test(model: str) -> int:
    raise SystemExit(0 if needs_tool_call_smoke_test(model) else 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRIM model profile helpers for shell scripts")
    parser.add_argument(
        "command",
        choices=("vllm-extra", "tool-parser", "needs-smoke-test"),
    )
    parser.add_argument("model", help="Model slug, HF id, or checkpoint path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "vllm-extra":
        return _cli_vllm_extra(args.model)
    if args.command == "tool-parser":
        return _cli_tool_parser(args.model)
    return _cli_needs_smoke_test(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
