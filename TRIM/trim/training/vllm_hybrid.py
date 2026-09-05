"""vLLM rollout + Transformers train hybrid (Scheme A).

vLLM owns batched generation (token IDs + sampled-token logprobs).
Transformers owns CISPO / sr_opd_ce backward, LoRA updates, and adapter IO.
The two engines never share GPUs: start vLLM → collect → stop vLLM →
start HF → one optimizer step → save adapter → repeat.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from trim.eval.harmony_runtime import (
    CANONICAL_STOP_TOKEN_IDS,
    O200K_HARMONY,
    decode_ids,
    fit_prompt_ids_to_context,
)
from trim.training.hf_rl_opd_client import CISPO_MAX_ACTION_TOKENS

WORKER_MODULE = "trim.training.vllm_rollout_worker"
FORBIDDEN_TOKENIZER_MARKERS = ("cl100k", "r50k", "p50k", "gpt2")
REQUIRED_SPECIAL_TOKEN_IDS = {"<|call|>": 200012, "<|return|>": 200002}
GPTOSS_VOCAB_SIZE = 201088


@dataclass
class GenerateRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int = 384
    temperature: float = 1.0
    seed: int = 0


@dataclass
class GenerateResult:
    request_id: str
    token_ids: list[int]
    token_logprobs: list[float]
    text: str
    logprob_old: float
    logprob_provenance: str
    action_mask: list[int] = field(default_factory=list)
    finish_reason: str = ""

    def __post_init__(self) -> None:
        if not self.action_mask:
            self.action_mask = [1] * len(self.token_ids)


def mean_behavior_logprob(
    token_logprobs: Sequence[float],
    *,
    max_action: int = CISPO_MAX_ACTION_TOKENS,
) -> float:
    """Mean sampled-token logprob on the same action window CISPO truncates to."""
    window = [float(x) for x in list(token_logprobs)[: max(1, int(max_action))]]
    if not window:
        return 0.0
    return float(sum(window) / len(window))


def cispo_row_from_generation(
    *,
    query_id: str,
    prompt_ids: Sequence[int],
    prompt_text: str,
    gen: GenerateResult,
    policy_version: str,
    turn_id: int,
    valid: bool,
) -> dict[str, Any]:
    """CISPO datum fields that must come from the behavior-policy sampler."""
    action_ids = list(gen.token_ids)
    mask = list(gen.action_mask or [1] * len(action_ids))
    if len(mask) != len(action_ids):
        mask = [1] * len(action_ids)
    return {
        "query_id": query_id,
        "prompt": prompt_text,
        "prompt_ids": list(prompt_ids),
        "action_text": gen.text,
        "action_ids": action_ids,
        "token_logprobs": list(gen.token_logprobs),
        "action_mask": mask,
        "logprob_old": float(gen.logprob_old),
        "logprob_provenance": gen.logprob_provenance,
        "n_tokens": len(action_ids),
        "policy_version": policy_version,
        "valid": valid,
        "turn_id": turn_id,
    }


def assert_gptoss_tokenizer(tokenizer: Any, *, source: str) -> dict[str, Any]:
    """Refuse cl100k_base / non-Harmony tokenizers. vLLM must use gpt-oss ids."""
    name = str(
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "name", None)
        or source
        or ""
    ).lower()
    for marker in FORBIDDEN_TOKENIZER_MARKERS:
        if marker in name:
            raise RuntimeError(
                f"tokenizer {name!r} looks like {marker}; "
                f"{O200K_HARMONY} / gpt-oss tokenizer is required"
            )
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    try:
        tokenizer_len = int(len(tokenizer) or 0)
    except (TypeError, AttributeError):
        tokenizer_len = 0
    effective_vocab_size = max(vocab_size, tokenizer_len)
    # Hugging Face reports the base BPE size (199998) for the official
    # gpt-oss tokenizer while Harmony control tokens are added above it.
    # Validate the effective vocabulary and canonical control-token IDs;
    # requiring config.vocab_size=201088 rejects valid Harness-1 checkpoints.
    if effective_vocab_size <= max(REQUIRED_SPECIAL_TOKEN_IDS.values()):
        raise RuntimeError(
            f"tokenizer effective_vocab_size={effective_vocab_size} cannot represent "
            "the canonical gpt-oss Harmony control tokens. cl100k_base fallback is forbidden."
        )
    resolved: dict[str, int] = {}
    for tok, expected in REQUIRED_SPECIAL_TOKEN_IDS.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        resolved[tok] = int(tid) if tid is not None else -1
        if resolved[tok] != expected:
            raise RuntimeError(
                f"tokenizer special {tok} id={resolved[tok]} != {expected}. "
                f"source={source}. This is not o200k_harmony/gpt-oss."
            )
    return {
        "encoding": O200K_HARMONY,
        "family": "gpt-oss",
        "source": source,
        "vocab_size": vocab_size,
        "effective_vocab_size": effective_vocab_size,
        "special_token_ids": resolved,
        "stop_token_ids": list(CANONICAL_STOP_TOKEN_IDS),
    }


def extract_sampled_logprobs(token_ids: Sequence[int], raw_logprobs: Any) -> list[float]:
    """Pull the sampled-token logprob at each position from a vLLM completion."""
    out: list[float] = []
    rows = list(raw_logprobs or [])
    for i, tid in enumerate(token_ids):
        if i >= len(rows) or rows[i] is None:
            out.append(0.0)
            continue
        slot = rows[i]
        logp = _logprob_for_token(slot, int(tid))
        out.append(0.0 if logp is None else float(logp))
    return out


def _logprob_for_token(slot: Any, token_id: int) -> float | None:
    if slot is None:
        return None
    if isinstance(slot, dict):
        if token_id in slot:
            return _coerce_logprob(slot[token_id])
        # vLLM sometimes keys decoded strings; take the sampled entry.
        for key, val in slot.items():
            if int(getattr(key, "real", key) if not isinstance(key, str) else -1) == token_id:
                return _coerce_logprob(val)
            inner_id = getattr(val, "token_id", None)
            if inner_id is not None and int(inner_id) == token_id:
                return _coerce_logprob(val)
        if len(slot) == 1:
            return _coerce_logprob(next(iter(slot.values())))
        return None
    if isinstance(slot, (list, tuple)) and slot:
        return _coerce_logprob(slot[0])
    return _coerce_logprob(slot)


def _coerce_logprob(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return None


def result_from_worker_row(row: dict[str, Any], *, enc) -> GenerateResult:
    token_ids = [int(x) for x in (row.get("token_ids") or [])]
    logprobs = [float(x) for x in (row.get("token_logprobs") or [])]
    if len(logprobs) != len(token_ids):
        logprobs = (logprobs + [0.0] * len(token_ids))[: len(token_ids)]
    return GenerateResult(
        request_id=str(row.get("request_id") or ""),
        token_ids=token_ids,
        token_logprobs=logprobs,
        text=decode_ids(enc, token_ids),
        logprob_old=mean_behavior_logprob(logprobs),
        logprob_provenance="vllm_sampled_token",
        action_mask=[1] * len(token_ids),
        finish_reason=str(row.get("finish_reason") or ""),
    )


def plan_cell_phases(
    cell: str,
    *,
    train_steps: int,
    on_policy_refresh: bool,
    use_frozen_states: bool,
) -> list[str]:
    """Scheme A phase list. RL / RL+OPD re-rollout after every optimizer step."""
    steps = max(0, int(train_steps))
    if cell == "before":
        return ["vllm_rollout", "vllm_eval"]
    if cell == "pure_opd" and use_frozen_states:
        return ["hf_train"] * steps + ["vllm_eval"]
    if cell in {"rl", "rl_opd"} and on_policy_refresh:
        phases: list[str] = []
        for _ in range(steps):
            phases.extend(["vllm_rollout", "hf_train"])
        phases.append("vllm_eval")
        return phases
    return ["vllm_rollout"] + ["hf_train"] * steps + ["vllm_eval"]


def default_tensor_parallel_size(explicit: int | None = None) -> int:
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    env = os.environ.get("SCAPE_VLLM_TP")
    if env:
        return max(1, int(env))
    try:
        import torch

        n = int(torch.cuda.device_count() or 0)
        return max(1, n)
    except Exception:
        return 1


class SchemeARuntime:
    """Mutually exclusive vLLM / HF occupancy of the same GPU set."""

    def __init__(self) -> None:
        self.hf: Any = None
        self.vllm: VLLMGenerateClient | None = None

    @property
    def vllm_active(self) -> bool:
        return self.vllm is not None and self.vllm.alive

    def assert_exclusive(self) -> None:
        if self.hf is not None and self.vllm_active:
            raise RuntimeError("scheme A violated: HF train and vLLM both resident")

    def attach_hf(self, backend: Any) -> Any:
        if self.vllm_active:
            raise RuntimeError("cannot load HF while vLLM holds the GPUs")
        self.hf = backend
        self.assert_exclusive()
        return backend

    def detach_hf(self) -> None:
        backend = self.hf
        self.hf = None
        if backend is None:
            return
        try:
            if getattr(backend, "model", None) is not None:
                del backend.model
            if getattr(backend, "optimizer", None) is not None:
                del backend.optimizer
        except Exception:
            pass
        _release_cuda()

    def attach_vllm(self, client: VLLMGenerateClient) -> VLLMGenerateClient:
        if self.hf is not None:
            raise RuntimeError("cannot start vLLM while HF train holds the GPUs")
        self.vllm = client
        self.assert_exclusive()
        return client

    def detach_vllm(self) -> None:
        client = self.vllm
        self.vllm = None
        if client is not None:
            client.close()
        _release_cuda()


def _release_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def wait_gpus_quiet(*, timeout_s: float = 8.0, max_bytes: int = 256 * 1024 * 1024) -> None:
    """Best-effort wait so the next engine can grab all cards.

    Keep this short. A 90s poll with the GPUs empty is exactly the idle
    window cluster watchdogs use to kill the job.
    """
    _release_cuda()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    deadline = time.time() + max(0.0, float(timeout_s))
    while True:
        used = []
        for i in range(torch.cuda.device_count()):
            try:
                used.append(int(torch.cuda.memory_allocated(i)))
            except Exception:
                used.append(0)
        if not used or all(u <= max_bytes for u in used):
            return
        if time.time() >= deadline:
            return
        time.sleep(0.2)
        _release_cuda()


class VLLMGenerateClient:
    """Parent-side handle to a long-lived vLLM worker subprocess.

    One worker session covers a whole multi-turn rollout (many generate_batch
    calls). Closing the worker frees every GPU before HF train starts.
    """

    def __init__(
        self,
        *,
        model_path: str,
        session_dir: Path,
        tensor_parallel_size: int,
        max_model_len: int = 8192,
        lora_path: str | None = None,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = True,
        python_exe: str | None = None,
        startup_timeout_s: float = 900.0,
        generate_timeout_s: float = 3600.0,
        extra_env: dict[str, str] | None = None,
        max_num_seqs: int | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.max_model_len = int(max_model_len)
        self.lora_path = str(Path(lora_path).resolve()) if lora_path else None
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enforce_eager = bool(enforce_eager)
        self.python_exe = python_exe or sys.executable
        self.startup_timeout_s = float(startup_timeout_s)
        self.generate_timeout_s = float(generate_timeout_s)
        self.extra_env = dict(extra_env or {})
        self.max_num_seqs = int(max_num_seqs) if max_num_seqs else None
        self.process: subprocess.Popen[bytes] | None = None
        self.tokenizer_audit: dict[str, Any] = {}
        self.n_generate_calls = 0
        self.n_prompts = 0

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def log_path(self) -> Path:
        return self.session_dir / "vllm_worker.log"

    def start(self) -> None:
        from trim.eval.model_tokenizer import encoding_config_for_model

        enc_cfg = encoding_config_for_model(self.model_path)
        cfg = {
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "stop_token_ids": list(enc_cfg["stop_token_ids"]),
            "encoding": enc_cfg["encoding"],
            "family": enc_cfg["family"],
            "max_num_seqs": self.max_num_seqs,
        }
        (self.session_dir / "config.json").write_text(
            json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("READY", "JOB", "RESULT", "SHUTDOWN"):
            path = self.session_dir / name
            if path.exists():
                path.unlink()
        env = os.environ.copy()
        env.update(self.extra_env)
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        trim_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = trim_root + os.pathsep + env.get("PYTHONPATH", "")
        log_fh = open(self.log_path, "w", encoding="utf-8")
        worker = Path(__file__).resolve().parent / "vllm_rollout_worker.py"
        cmd = [
            self.python_exe,
            str(worker),
            "--session-dir",
            str(self.session_dir),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        self._wait_flag("READY", self.startup_timeout_s, what="vLLM worker start")
        audit_path = self.session_dir / "tokenizer_audit.json"
        if audit_path.is_file():
            self.tokenizer_audit = json.loads(audit_path.read_text(encoding="utf-8"))

    def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResult]:
        if not self.alive:
            raise RuntimeError("vLLM worker is not running")
        from trim.eval.model_tokenizer import assert_family_prompt_ids, encoding_config_for_model

        audit = self.tokenizer_audit or {}
        enc_cfg = encoding_config_for_model(self.model_path)
        family = str(audit.get("family") or enc_cfg["family"])
        prompt_rows = []
        for req in requests:
            prompt_ids = fit_prompt_ids_to_context(
                req.prompt_token_ids,
                max_model_len=self.max_model_len,
                max_new_tokens=req.max_new_tokens,
            )
            prompt_ids = assert_family_prompt_ids(
                prompt_ids,
                family=family,
                what=f"vLLM prompt {req.request_id}",
            )
            prompt_rows.append(
                asdict(
                    GenerateRequest(
                        request_id=req.request_id,
                        prompt_token_ids=prompt_ids,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        seed=req.seed,
                    )
                )
            )
        payload = {"cmd": "generate", "requests": prompt_rows}
        (self.session_dir / "job.json").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
        result_flag = self.session_dir / "RESULT"
        if result_flag.exists():
            result_flag.unlink()
        (self.session_dir / "JOB").write_text("1\n", encoding="utf-8")
        self._wait_flag("RESULT", self.generate_timeout_s, what="vLLM generate")
        blob = json.loads((self.session_dir / "result.json").read_text(encoding="utf-8"))
        if blob.get("error"):
            raise RuntimeError(f"vLLM generate failed: {blob['error']}\n{_tail(self.log_path)}")
        decoder = self._decoder()
        by_id = {str(row["request_id"]): row for row in blob.get("outputs") or []}
        ordered: list[GenerateResult] = []
        for req in requests:
            row = by_id.get(req.request_id)
            if row is None:
                raise RuntimeError(f"vLLM missing output for request_id={req.request_id}")
            result = result_from_worker_row(row, enc=decoder)
            if row.get("text"):
                result.text = str(row["text"])
            ordered.append(result)
        self.n_generate_calls += 1
        self.n_prompts += len(requests)
        return ordered

    def _decoder(self):
        enc = getattr(self, "_model_enc", None)
        if enc is None:
            from trim.eval.model_tokenizer import load_model_encoding

            enc = load_model_encoding(self.model_path)
            self._model_enc = enc
        return enc

    def close(self) -> None:
        proc = self.process
        if proc is None:
            return
        try:
            (self.session_dir / "SHUTDOWN").write_text("1\n", encoding="utf-8")
        except Exception:
            pass
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.process = None

    def _wait_flag(self, name: str, timeout_s: float, *, what: str) -> None:
        flag = self.session_dir / name
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"{what}: worker exited with code {self.process.returncode}.\n"
                    f"Log tail:\n{_tail(self.log_path)}"
                )
            if flag.is_file():
                return
            time.sleep(0.25)
        raise TimeoutError(f"{what} timed out after {timeout_s}s.\nLog tail:\n{_tail(self.log_path)}")


def _tail(path: Path, n: int = 60) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


class HFGenerateClient:
    """Sequential Transformers generate() fallback. Same GenerateResult schema."""

    def __init__(self, backend: Any, *, enc, logprob_from_hf: bool = True) -> None:
        self.backend = backend
        self.enc = enc
        self.logprob_from_hf = bool(logprob_from_hf)
        self.logprob_provenance = "hf_teacher_forced" if logprob_from_hf else "none"

    def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResult]:
        from trim.eval.model_tokenizer import assert_family_prompt_ids
        from trim.training.four_cell_runtime import generate_harmony

        family = str(getattr(self.enc, "family", "") or "")
        out: list[GenerateResult] = []
        for req in requests:
            prompt_ids = assert_family_prompt_ids(
                req.prompt_token_ids,
                family=family,
                what=f"HF prompt {req.request_id}",
            )
            gen = generate_harmony(
                self.backend,
                "",
                enc=self.enc,
                max_new=req.max_new_tokens,
                sample=req.temperature > 0,
                seed=req.seed,
                prompt_ids=list(prompt_ids),
            )
            token_ids = list(gen["action_ids"])
            logprobs: list[float] = []
            logprob_old = 0.0
            if self.logprob_from_hf and token_ids:
                import torch

                old_prompt = prompt_ids[-384:] if len(prompt_ids) > 384 else prompt_ids
                old_act = token_ids[:CISPO_MAX_ACTION_TOKENS]
                with torch.no_grad():
                    old_lp = self.backend._teacher_forced_logprobs(
                        old_prompt, old_act, require_grad=False
                    )
                logprobs = [float(x) for x in old_lp.detach().cpu().tolist()]
                logprob_old = mean_behavior_logprob(logprobs)
            out.append(
                GenerateResult(
                    request_id=req.request_id,
                    token_ids=token_ids,
                    token_logprobs=logprobs,
                    text=str(gen.get("text") or ""),
                    logprob_old=logprob_old,
                    logprob_provenance=self.logprob_provenance,
                    action_mask=[1] * len(token_ids),
                )
            )
        return out


def load_adapter_weights(backend: Any, adapter_dir: Path | str | None) -> dict[str, Any]:
    """Reload a saved LoRA onto a freshly constructed PEFT model (theta0 or cell)."""
    if not adapter_dir:
        return {"loaded": False, "adapter_dir": None}
    path = Path(adapter_dir)
    weight_file = path / "adapter_model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(f"adapter missing: {weight_file}")
    from safetensors.torch import load_file
    from trim.eval.adapter_reload_audit import remap_lora_state

    weights = remap_lora_state(load_file(str(weight_file)))
    missing, unexpected = backend.model.load_state_dict(weights, strict=False)
    lora_missing = [x for x in missing if "lora_" in x]
    if lora_missing:
        raise RuntimeError(f"adapter reload failed: {lora_missing[:8]}")
    return {
        "loaded": True,
        "adapter_dir": str(path),
        "unexpected_lora": [x for x in unexpected if "lora_" in x],
    }


def materialize_vllm_base(
    *,
    base_model: str,
    sft_adapter: str,
    cache_dir: Path,
    device_map: str,
) -> str:
    """vLLM needs a real HF directory. Merge Clean-SFT once if the launcher passed an adapter."""
    adapter = Path(sft_adapter) if sft_adapter else None
    if adapter is None or not adapter.exists() or not (adapter / "adapter_config.json").is_file():
        return base_model
    marker = cache_dir / "config.json"
    if marker.is_file():
        return str(cache_dir)
    from trim.training.hf_tool_opd import ScapeHFToolOPD

    cache_dir.mkdir(parents=True, exist_ok=True)
    backend = ScapeHFToolOPD(
        model_path=str(adapter),
        device_map=device_map,
        use_lora=False,
    )
    try:
        backend.model.save_pretrained(str(cache_dir))
        backend.tokenizer.save_pretrained(str(cache_dir))
    finally:
        del backend.model
        _release_cuda()
    return str(cache_dir)
