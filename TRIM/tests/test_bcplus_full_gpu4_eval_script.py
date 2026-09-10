"""Sanity checks for bcplus_full GPU4 launch script."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_bcplus_full_gpu4_eval.sh"


def _bash_fn(name: str, *args: str) -> str:
    cmd = f'source "{SCRIPT}" >/dev/null 2>&1; {name} {" ".join(args)}'
    out = subprocess.check_output(["bash", "-c", cmd], text=True)
    return out.strip()


def test_vllm_extra_qwen_uses_hermes_parser():
    extra = _bash_fn("vllm_extra_for_model", "Qwen3-4B-Instruct-2507")
    assert "--tool-call-parser hermes" in extra
    assert "openai" not in extra.split("--tool-call-parser")[1].split()[0]


def test_vllm_extra_harness_uses_openai_parser():
    extra = _bash_fn("vllm_extra_for_model", "harness-1")
    assert "--tool-call-parser openai" in extra


def test_normalize_pid_strips_log_noise():
    out = subprocess.check_output(
        [
            "bash",
            "-c",
            f'source "{SCRIPT}" >/dev/null 2>&1; normalize_pid $\'[2026-09-10T07:16:00+08:00] starting vLLM\\n12345\\n\'',
        ],
        text=True,
    )
    assert out.strip() == "12345"
