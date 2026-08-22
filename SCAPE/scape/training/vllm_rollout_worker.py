#!/usr/bin/env python3
"""Long-lived vLLM worker for Scheme A batched Harmony rollouts.

Reads session_dir/config.json, loads the gpt-oss tokenizer (never cl100k),
then waits for JOB / SHUTDOWN flags. Each JOB is a batch of prompt_token_ids.
Returns sampled token IDs and per-token logprobs so CISPO can be constructed
without a second HF generate() pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCAPE = Path(__file__).resolve().parents[2]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))


def _wait_flag(path: Path, *, shutdown: Path, interval: float = 0.1) -> str:
    while True:
        if shutdown.is_file():
            return "shutdown"
        if path.is_file():
            return "job"
        time.sleep(interval)


def _tokens_prompt(token_ids: list[int]) -> Any:
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=list(token_ids))
    except Exception:
        return {"prompt_token_ids": list(token_ids)}


def _sampling_params(req: dict[str, Any], stop_token_ids: list[int]):
    from vllm import SamplingParams

    temperature = float(req.get("temperature") or 0.0)
    kwargs: dict[str, Any] = {
        "temperature": max(0.0, temperature),
        "max_tokens": int(req.get("max_new_tokens") or 384),
        "stop_token_ids": list(stop_token_ids),
        "logprobs": 1,
        "include_stop_str_in_output": True,
    }
    if temperature <= 0:
        kwargs["temperature"] = 0.0
    else:
        kwargs["seed"] = int(req.get("seed") or 0)
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("include_stop_str_in_output", None)
        return SamplingParams(**kwargs)


def _completion_token_ids(output: Any) -> list[int]:
    comp = output.outputs[0]
    ids = getattr(comp, "token_ids", None) or getattr(comp, "output_token_ids", None)
    return [int(x) for x in (ids or [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM Harmony rollout worker")
    parser.add_argument("--session-dir", type=Path, required=True)
    args = parser.parse_args()
    session = args.session_dir
    session.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((session / "config.json").read_text(encoding="utf-8"))

    from scape.eval.harmony_runtime import CANONICAL_STOP_TOKEN_IDS, O200K_HARMONY
    from scape.training.vllm_hybrid import (
        assert_gptoss_tokenizer,
        extract_sampled_logprobs,
    )

    stop_token_ids = [int(x) for x in (cfg.get("stop_token_ids") or CANONICAL_STOP_TOKEN_IDS)]
    if stop_token_ids != list(CANONICAL_STOP_TOKEN_IDS):
        raise SystemExit(
            f"worker stop_token_ids={stop_token_ids} != {CANONICAL_STOP_TOKEN_IDS}"
        )
    encoding = str(cfg.get("encoding") or "")
    if encoding and encoding != O200K_HARMONY:
        raise SystemExit(f"worker encoding={encoding} != {O200K_HARMONY}")

    from transformers import AutoTokenizer

    model_path = str(cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    audit = assert_gptoss_tokenizer(tokenizer, source=model_path)
    (session / "tokenizer_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": int(cfg.get("tensor_parallel_size") or 1),
        "max_model_len": int(cfg.get("max_model_len") or 8192),
        "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization") or 0.90),
        "dtype": "auto",
        "disable_log_stats": True,
    }
    if cfg.get("enforce_eager", True):
        llm_kwargs["enforce_eager"] = True
    lora_path = cfg.get("lora_path") or None
    lora_request = None
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_lora_rank"] = 16
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("policy", 1, str(lora_path))

    print(json.dumps({"event": "loading_llm", **{k: llm_kwargs[k] for k in ("model", "tensor_parallel_size", "max_model_len") if k in llm_kwargs}, "lora": lora_path}), flush=True)
    llm = LLM(**llm_kwargs)
    print(json.dumps({"event": "llm_ready"}), flush=True)
    (session / "READY").write_text("1\n", encoding="utf-8")

    job_flag = session / "JOB"
    shutdown_flag = session / "SHUTDOWN"
    while True:
        kind = _wait_flag(job_flag, shutdown=shutdown_flag)
        if kind == "shutdown":
            break
        job = json.loads((session / "job.json").read_text(encoding="utf-8"))
        if job_flag.exists():
            job_flag.unlink()
        try:
            requests = list(job.get("requests") or [])
            prompts = [_tokens_prompt([int(x) for x in (r.get("prompt_token_ids") or [])]) for r in requests]
            params = [_sampling_params(r, stop_token_ids) for r in requests]
            generate_kwargs: dict[str, Any] = {}
            if lora_request is not None:
                generate_kwargs["lora_request"] = lora_request
            outputs = llm.generate(prompts, params, **generate_kwargs)
            rows = []
            for req, output in zip(requests, outputs):
                token_ids = _completion_token_ids(output)
                raw_lp = getattr(output.outputs[0], "logprobs", None)
                token_logprobs = extract_sampled_logprobs(token_ids, raw_lp)
                rows.append(
                    {
                        "request_id": req.get("request_id"),
                        "token_ids": token_ids,
                        "token_logprobs": token_logprobs,
                        "finish_reason": getattr(output.outputs[0], "finish_reason", None),
                    }
                )
            (session / "result.json").write_text(
                json.dumps({"ok": True, "outputs": rows}) + "\n", encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            (session / "result.json").write_text(
                json.dumps({"ok": False, "error": repr(exc), "outputs": []}) + "\n",
                encoding="utf-8",
            )
        (session / "RESULT").write_text("1\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
