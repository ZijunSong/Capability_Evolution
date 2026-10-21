"""Single-device BF16 load for pure DDP LoRA. FSDP/ZeRO use their own loaders."""

from __future__ import annotations

import hashlib
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any


def make_accelerator():
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            ),
            InitProcessGroupKwargs(timeout=timedelta(minutes=30)),
        ],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("SFT v2 requires a CUDA training device")
    if not torch.cuda.is_available():
        raise RuntimeError("SFT v2 requires CUDA")
    return accelerator


def load_ddp_model(model_path: str, adapter_path: str, accelerator: Any, attn_impl: str | None = None) -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, Mxfp4Config

    if not adapter_path:
        raise ValueError("Main ablation requires the common initial adapter")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    kwargs: dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": str(accelerator.device)},
    )
    quant = getattr(config, "quantization_config", None)
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    if isinstance(quant, dict) and quant.get("quant_method") == "mxfp4":
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    elif quant:
        raise ValueError("Unvalidated quantization config for BF16 SFT")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    model.config.output_router_logits = False
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    stray = [(name, str(param.device)) for name, param in model.named_parameters() if param.device != accelerator.device]
    if stray:
        raise RuntimeError(f"Parameters outside the training device: {stray[:8]}")
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    meta = [name for name, param in model.named_parameters() if getattr(param.device, "type", "") == "meta"]
    if meta:
        raise RuntimeError(f"Parameters left on meta: {meta[:8]}")
    return model


def probe_action_logits(model: Any, device: Any) -> None:
    """Fail before training if the model ignores tensor ``logits_to_keep``."""
    import torch

    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    if vocab < 4:
        embedding = model.get_input_embeddings()
        vocab = int(getattr(embedding, "num_embeddings", 0) or 0)
    if vocab < 4:
        raise RuntimeError("cannot probe logits_to_keep without a vocab size")
    token = 1 if vocab > 2 else 0
    inputs = torch.tensor([[token, token, token]], dtype=torch.long, device=device)
    positions = torch.tensor([1], dtype=torch.long, device=device)
    outputs = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        use_cache=False,
        output_router_logits=False,
        logits_to_keep=positions,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if tuple(logits.shape[:2]) != (1, 1):
        raise RuntimeError(
            "Model did not honor action-position logits selection during the startup probe; "
            "refusing to continue with full-sequence logits"
        )


def device_uuid(device: Any) -> str:
    import torch

    if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(device)
    value = getattr(props, "uuid", "") or ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value or "unknown")


def placement_record(accelerator: Any) -> dict[str, Any]:
    import os

    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": int(accelerator.process_index),
        "local_rank": int(getattr(accelerator, "local_process_index", 0)),
        "world_size": int(accelerator.num_processes),
        "device": str(accelerator.device),
        "gpu_uuid": device_uuid(accelerator.device),
    }


def assert_distinct_gpu_uuids(accelerator: Any, records: list[dict[str, Any]]) -> None:
    del accelerator
    by_host: dict[str, list[str]] = {}
    for item in records:
        by_host.setdefault(str(item["hostname"]), []).append(str(item["gpu_uuid"]))
    for host, uuids in by_host.items():
        if any(item in {"", "unknown", "cpu"} for item in uuids):
            raise RuntimeError(f"training rank on {host} has no CUDA device UUID: {uuids}")
        if len(uuids) != len(set(uuids)):
            raise RuntimeError(f"training ranks on {host} share a GPU UUID: {uuids}")


def kernel_report(model: Any) -> dict[str, Any]:
    import inspect

    names = sorted({type(module).__name__ for module in model.modules()})
    dtypes: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.dtype)
        dtypes[key] = dtypes.get(key, 0) + int(param.numel())
    config = getattr(model, "config", None)
    base = getattr(model, "get_base_model", lambda: model)()
    try:
        source = inspect.getfile(type(base))
    except Exception:
        source = ""
    source_sha = ""
    if source and Path(source).is_file():
        digest = hashlib.sha256()
        with open(source, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_sha = digest.hexdigest()
    quant = getattr(config, "quantization_config", None)
    return {
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "attention_modules": [name for name in names if "Attn" in name or "Attention" in name],
        "expert_modules": [name for name in names if "Expert" in name or "Moe" in name or "MoE" in name],
        "quantized": quant is not None,
        "dtypes": dtypes,
        "implementation_file": source,
        "implementation_sha256": source_sha,
    }


def trainable_checksum(model: Any) -> str:
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(param.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_optimizer(model: Any, args: Any):
    import torch

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return optimizer, scheduler
