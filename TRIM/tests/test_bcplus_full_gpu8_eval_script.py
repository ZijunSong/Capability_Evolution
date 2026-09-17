"""Sanity checks for bcplus GPU8 launch script vLLM flags."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


LIB = Path(__file__).resolve().parents[1] / "scripts" / "lib_bcplus_w_harness1_gpu8_eval.sh"
VLLM_EXTRA = Path(__file__).resolve().parents[1] / "scripts" / "vllm_model_extra.sh"


def _bash_vllm_extra(slug: str) -> str:
    cmd = (
        f'export SCRIPT_DIR="{LIB.parent}" TRIM_ROOT="{LIB.parent.parent}" PY="{sys.executable}"; '
        f'source "{VLLM_EXTRA}"; vllm_extra_for_model "{slug}"'
    )
    return subprocess.check_output(["bash", "-c", cmd], text=True).strip()


def test_vllm_extra_qwen_uses_hermes_parser():
    extra = _bash_vllm_extra("Qwen3-4B-Instruct-2507")
    assert "--tool-call-parser hermes" in extra
    assert "openai" not in extra.split("--tool-call-parser")[1].split()[0]


def test_vllm_extra_harness_uses_openai_parser():
    extra = _bash_vllm_extra("harness-1")
    assert "--tool-call-parser openai" in extra


def test_vllm_extra_glm_uses_glm45_parser():
    extra = _bash_vllm_extra("glm-4-9b-chat")
    assert "--tool-call-parser glm45" in extra


def test_vllm_extra_glm0414_uses_plugin_parser():
    extra = _bash_vllm_extra("GLM-4-32B-0414")
    assert "--tool-call-parser glm4_0414" in extra
    assert "--tool-parser-plugin" in extra
    assert "glm4_0414_tool_parser.py" in extra


def test_normalize_pid_strips_log_noise():
    out = subprocess.check_output(
        [
            "bash",
            "-c",
            f'export SCRIPT_DIR="{LIB.parent}"; source "{LIB}" >/dev/null 2>&1; '
            f"normalize_pid $'[2026-09-10T07:16:00+08:00] starting vLLM\\n12345\\n'",
        ],
        text=True,
    )
    assert out.strip() == "12345"
