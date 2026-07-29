"""Manage a local vLLM actor server subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import IO

from training.opd.vllm_rollout_backend import wait_for_vllm_server


@dataclass
class VLLMServerHandle:
    process: subprocess.Popen[bytes]
    base_url: str
    log_file: IO[str] | None = None

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if self.log_file is not None:
            self.log_file.close()


def _tail_log(path: str | None, n: int = 40) -> str:
    if not path or not os.path.exists(path):
        return ""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    return "\n".join(lines[-n:])


def start_vllm_server(
    *,
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    tensor_parallel_size: int = 4,
    max_model_len: int = 8192,
    served_model_name: str = "qwen",
    log_path: str | None = None,
    extra_env: dict[str, str] | None = None,
    enforce_eager: bool = True,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str | None = None,
) -> VLLMServerHandle:
    """Launch `vllm serve` and wait until the OpenAI API is ready."""
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "bfloat16",
        "--disable-custom-all-reduce",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if enable_auto_tool_choice:
        cmd.append("--enable-auto-tool-choice")
        if tool_call_parser:
            cmd.extend(["--tool-call-parser", tool_call_parser])

    env = os.environ.copy()
    # Smoke-friendly defaults; override in production if needed.
    env.setdefault("VLLM_USE_V1", "0")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    if extra_env:
        env.update(extra_env)

    log_fh = open(log_path, "w", encoding="utf-8") if log_path else None
    process = subprocess.Popen(
        cmd,
        stdout=log_fh or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
        env=env,
    )
    base_url = f"http://{host}:{port}/v1"
    deadline = time.time() + 900.0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited with code {process.returncode}.\n"
                f"Log tail:\n{_tail_log(log_path)}"
            )
        try:
            wait_for_vllm_server(base_url, timeout_s=5.0, poll_interval_s=1.0)
            time.sleep(1.0)
            return VLLMServerHandle(process=process, base_url=base_url, log_file=log_fh)
        except TimeoutError:
            time.sleep(2.0)
    raise TimeoutError(
        f"vLLM server not ready at {base_url} after 900s.\n"
        f"Log tail:\n{_tail_log(log_path)}"
    )
