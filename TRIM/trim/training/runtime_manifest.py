"""Low-overhead step timing plus the actor runtime manifest.

Phase timers are wall-clock accumulators. They are not per-token CUDA events.
``unattributed_s`` is whatever remains after the disjoint phase list below.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

PARTITION_KEYS: tuple[str, ...] = (
    "sample_broadcast_s",
    "phase_switch_s",
    "rollout_engine_start_s",
    "rollout_generate_s",
    "rollout_engine_close_s",
    "gather_s",
    "teacher_projector_build_s",
    "batch_plan_s",
    "actor_load_s",
    "teacher_score_s",
    "student_forward_s",
    "loss_s",
    "backward_s",
    "grad_sync_s",
    "optimizer_s",
    "actor_update_other_s",
    "adapter_save_s",
    "optimizer_save_s",
    "adapter_publish_s",
    "barrier_wait_s",
)

_SOURCE_FILES: tuple[str, ...] = (
    "trim/training/tinker_opd_datum.py",
    "trim/training/tinker_rl_opd_trainer.py",
    "trim/training/batched_env_rollout.py",
    "trim/training/hf_rl_batch.py",
    "trim/training/dist_runtime.py",
    "trim/integrations/verl/trainer_adapter.py",
    "trim/integrations/verl/fsdp2_actor.py",
)

_IMPORT_MODULES: tuple[str, ...] = (
    "trim.integrations.verl.trainer_adapter",
    "trim.integrations.verl.fsdp2_actor",
    "trim.training.tinker_opd_datum",
    "trim.training.hf_rl_batch",
    "trim.training.batched_env_rollout",
)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(root: Path, args: Sequence[str], *, limit: int = 4000) -> str:
    try:
        text = subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return ""
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "\n…"
    return text


_FINGERPRINTS: dict[str, dict[str, Any]] = {}


def source_fingerprint(root: Path | None = None) -> dict[str, Any]:
    base = Path(root or _package_root())
    cached = _FINGERPRINTS.get(str(base))
    if cached is not None:
        return dict(cached)
    git_root = base
    for candidate in [base, *base.parents]:
        if (candidate / ".git").exists():
            git_root = candidate
            break
    files: dict[str, str] = {}
    for rel in _SOURCE_FILES:
        path = base / rel
        if path.is_file():
            files[rel] = _sha256(path)
    dirty = _git_text(git_root, ["status", "--porcelain"], limit=8000)
    payload = {
        "package_root": str(base),
        "git_root": str(git_root),
        "commit": _git_text(git_root, ["rev-parse", "HEAD"], limit=80),
        "dirty": bool(dirty),
        "dirty_stat": _git_text(git_root, ["diff", "--stat"], limit=4000),
        "source_sha256": files,
    }
    _FINGERPRINTS[str(base)] = payload
    return dict(payload)


def _module_file(name: str) -> str:
    try:
        import importlib

        module = importlib.import_module(name)
    except Exception as exc:
        return f"import_failed:{type(exc).__name__}"
    try:
        return str(inspect.getfile(module))
    except Exception:
        return "unknown"


def _package_version(name: str) -> str:
    try:
        import importlib.metadata as metadata

        return str(metadata.version(name))
    except Exception:
        return ""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            return str(value)
    return str(value)


def _unwrap_model(model: Any) -> Any:
    core = model
    seen: set[int] = set()
    while hasattr(core, "module") and id(core) not in seen:
        seen.add(id(core))
        nxt = getattr(core, "module")
        if nxt is None or nxt is core:
            break
        core = nxt
    return core


def collect_actor_runtime(actor: Any, *, requested_dtype: str = "bfloat16") -> dict[str, Any]:
    """Record the actor that actually loaded, including a logged quantization fallback."""
    import torch

    core = _unwrap_model(getattr(actor, "model", None))
    expert_classes: set[str] = set()
    attention_classes: set[str] = set()
    quantizer = ""
    attn_impl = ""
    quant_config: Any = None
    dtypes: dict[str, dict[str, int]] = {}
    trainable = 0
    total = 0
    if core is not None:
        quantizer_obj = getattr(core, "hf_quantizer", None)
        if quantizer_obj is not None:
            quantizer = type(quantizer_obj).__name__
        config = getattr(core, "config", None)
        if config is not None:
            attn_impl = str(
                getattr(config, "_attn_implementation", None)
                or getattr(config, "attn_implementation", None)
                or ""
            )
            quant_config = getattr(config, "quantization_config", None)
        for module in core.modules():
            name = type(module).__name__
            lowered = name.lower()
            if "expert" in lowered or "moe" in lowered:
                expert_classes.add(name)
            if "attention" in lowered or lowered.endswith("attn"):
                attention_classes.add(name)
        for param in core.parameters():
            key = str(param.dtype).replace("torch.", "")
            bucket = dtypes.setdefault(key, {"numel": 0, "bytes": 0, "trainable_numel": 0})
            numel = int(param.numel())
            bucket["numel"] += numel
            bucket["bytes"] += numel * int(param.element_size())
            total += numel
            if param.requires_grad:
                trainable += numel
                bucket["trainable_numel"] += numel
    allocated = reserved = None
    capability = ""
    device_name = ""
    if torch.cuda.is_available():
        try:
            device = getattr(actor, "device", None)
            index = device.index if device is not None and getattr(device, "type", "") == "cuda" else torch.cuda.current_device()
            allocated = round(float(torch.cuda.memory_allocated(index)) / (1024 * 1024), 1)
            reserved = round(float(torch.cuda.memory_reserved(index)) / (1024 * 1024), 1)
            major, minor = torch.cuda.get_device_capability(index)
            capability = f"{major}.{minor}"
            device_name = str(torch.cuda.get_device_name(index))
        except Exception as exc:
            device_name = f"cuda_query_failed:{type(exc).__name__}"
    quant_present = quant_config not in (None, {}, "")
    packed = any("int" in key or "float8" in key or "uint" in key for key in dtypes)
    fallback = ""
    if quant_present and not packed and dtypes:
        fallback = (
            "quantization_config is present but loaded parameters are not packed; "
            f"requested_dtype={requested_dtype}. This is the effective training dtype, "
            "not proof that vLLM MXFP4 kernels are used for backward."
        )
    versions = {
        "torch": getattr(torch, "__version__", ""),
        "transformers": _package_version("transformers"),
        "vllm": _package_version("vllm"),
        "peft": _package_version("peft"),
        "kernels": _package_version("kernels"),
        "triton": _package_version("triton"),
    }
    payload = {
        "requested_dtype": str(requested_dtype),
        "effective_parameter_dtypes": dtypes,
        "trainable_params": int(trainable),
        "total_params_seen": int(total),
        "expert_classes": sorted(expert_classes),
        "attention_classes": sorted(attention_classes),
        "attention_implementation": attn_impl,
        "quantizer_class": quantizer,
        "quantization_config": _jsonable(quant_config) if quant_present else None,
        "quantization_fallback": fallback,
        "wrap": str(getattr(actor, "wrap", "") or ""),
        "wrapper_type": str(getattr(actor, "wrapper_type", "") or ""),
        "model_path": str(getattr(actor, "model_path", "") or ""),
        "rank": int(getattr(actor, "rank", 0) or 0),
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "gpu_capability": capability,
        "gpu_name": device_name,
        "versions": versions,
        "imports": {name: _module_file(name) for name in _IMPORT_MODULES},
        "source": source_fingerprint(),
        "pid": os.getpid(),
    }
    if fallback:
        print(json.dumps({"event": "actor_quantization_fallback", "detail": fallback, "dtypes": dtypes}), flush=True)
    return payload


def finalize_step_timing(marks: Mapping[str, Any]) -> dict[str, float]:
    """Disjoint phase times plus the residual ``unattributed_s``."""
    out: dict[str, float] = {}
    for key, value in marks.items():
        if str(key).startswith("_"):
            continue
        try:
            out[str(key)] = round(float(value or 0.0), 3)
        except (TypeError, ValueError):
            continue
    for key in PARTITION_KEYS:
        out.setdefault(key, 0.0)
    wall = float(marks.get("_wall") or out.get("step_wall_s") or 0.0)
    accounted = sum(float(out.get(key) or 0.0) for key in PARTITION_KEYS)
    out["step_wall_s"] = round(wall, 3)
    out["unattributed_s"] = round(max(0.0, wall - accounted), 3)
    return out


def summarize_rank_timings(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        vals: list[float] = []
        for row in rows:
            try:
                vals.append(float(row.get(key) or 0.0))
            except (TypeError, ValueError):
                vals.append(0.0)
        if not vals:
            continue
        summary[key] = {
            "min": round(min(vals), 3),
            "mean": round(sum(vals) / len(vals), 3),
            "max": round(max(vals), 3),
        }
    return summary
