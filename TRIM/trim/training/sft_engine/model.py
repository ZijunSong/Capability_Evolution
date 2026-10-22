"""Single-device BF16 load for pure DDP LoRA. FSDP/ZeRO use their own loaders."""

from __future__ import annotations

import hashlib
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any

# Decoder-layer class names accepted by Accelerate's transformer_based_wrap.
# First FSDP2 version is validated for GPT-OSS only; refuse anything else
# instead of silently wrapping the whole model in one shard group.
FSDP2_TRANSFORMER_CLS: dict[str, tuple[str, ...]] = {
    "gpt_oss": ("GptOssDecoderLayer",),
}


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


def load_ddp_model(
    model_path: str,
    adapter_path: str,
    accelerator: Any,
    attn_impl: str | None = None,
    gradient_checkpointing: bool = False,
) -> Any:
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
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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


def fsdp2_wrap_classes(model_type: str) -> tuple[str, ...]:
    names = FSDP2_TRANSFORMER_CLS.get(str(model_type or ""))
    if not names:
        raise ValueError(
            f"accelerate_fsdp2 first implementation is validated for GPT-OSS only, got model_type={model_type!r}"
        )
    return names


def make_fsdp2_accelerator() -> Any:
    """Programmatic FSDP2 plugin; the outer launcher still owns torchrun.

    ``cpu_ram_efficient_loading=True`` also sets the transformers env flag so
    non-main ranks do not materialize the full BF16 base in CPU RAM. The plugin
    must therefore be constructed BEFORE ``from_pretrained``.
    """
    import torch
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import InitProcessGroupKwargs
    from torch.distributed.fsdp import MixedPrecisionPolicy

    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        auto_wrap_policy="transformer_based_wrap",
        state_dict_type="SHARDED_STATE_DICT",
        cpu_ram_efficient_loading=True,
        cpu_offload=False,
        activation_checkpointing=False,
        mixed_precision_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )
    acc = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        fsdp_plugin=plugin,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    if acc.device.type != "cuda" or not getattr(acc, "is_fsdp2", False):
        raise RuntimeError("FSDP2 CUDA backend was not activated")
    torch.cuda.set_device(acc.device)
    return acc


def load_fsdp2_model(
    model_path: str,
    adapter_path: str,
    accelerator: Any,
    attn_impl: str | None = None,
    gradient_checkpointing: str = "off",
) -> Any:
    """CPU/meta load of the BF16 base plus the common initial LoRA, no ``device_map``.

    Must run after :func:`make_fsdp2_accelerator` (plugin construction sets the
    RAM-efficient loading env) and before ``accelerator.prepare``. Never call
    ``load_ddp_model`` here: its ``device_map={"": cuda}`` defeats sharding.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    if not adapter_path:
        raise ValueError("Main ablation requires the common initial adapter")
    if not getattr(accelerator, "is_fsdp2", False):
        raise RuntimeError("load_fsdp2_model requires the FSDP2 accelerator")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    quant = getattr(config, "quantization_config", None)
    if quant is not None:
        raise ValueError("accelerate_fsdp2 requires the prepared BF16 base, not a quantized checkpoint")
    wrap_classes = fsdp2_wrap_classes(getattr(config, "model_type", None))
    kwargs: dict[str, Any] = dict(
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    # No device_map and no .to(cuda) here: prepare shards from CPU/meta.
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    try:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True, torch_device="cpu")
    except TypeError as exc:
        if "torch_device" not in str(exc):
            raise
        # Locked PEFT without torch_device support: default keeps tensors on CPU
        # for a model that is already on CPU, which is what we need.
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.config.use_cache = False
    model.config.output_router_logits = False
    mode = str(gradient_checkpointing or "off")
    if mode == "non_reentrant":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif mode != "off":
        raise ValueError(f"accelerate_fsdp2 accepts only off or non_reentrant, got {mode}")
    model.train()
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not params or any("lora_" not in name for name, _ in params):
        raise RuntimeError("expected the common attention-only LoRA trainables")
    stray = [
        (name, str(param.device))
        for name, param in model.named_parameters()
        if getattr(param.device, "type", "") not in {"cpu", "meta"}
    ]
    if stray:
        raise RuntimeError(f"parameters left the CPU load stage early: {stray[:8]}")
    # NOTE: with ACCELERATE_USE_FSDP + FSDP_CPU_RAM_EFFICIENT_LOADING,
    # transformers 4.57.6 loads non-main ranks on meta
    # (modeling_utils: `device = "meta" if is_fsdp_enabled() and not
    # is_local_dist_rank_0() else "cpu"`). That is the intended RAM-efficient
    # state; prepare broadcasts rank 0's weights into the shards. Only a
    # *distributed-free* run with meta params is a broken load here.
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        meta = [name for name, param in model.named_parameters() if getattr(param.device, "type", "") == "meta"]
        if meta:
            raise RuntimeError(f"meta parameters without a distributed load contract: {meta[:8]}")
    decoder_names = collect_decoder_module_names(model, wrap_classes)
    model._trim_fsdp2_decoder_names = decoder_names
    model._trim_fsdp2_wrap_classes = wrap_classes
    return model


def collect_decoder_module_names(model: Any, wrap_classes: tuple[str, ...]) -> set[str]:
    """Qualified names of the decoder layers, registered BEFORE prepare.

    FSDP2 rewrites module classes at prepare time, so after sharding the
    original ``type(m).__name__`` check is unreliable; keep the paths instead.
    """
    names: set[str] = set()
    wanted = set(wrap_classes)
    for name, module in model.named_modules():
        if type(module).__name__ in wanted:
            names.add(name)
    if not names:
        raise RuntimeError(f"model has no modules named {sorted(wanted)}; fix transformer_cls_names_to_wrap")
    return names


def assert_fsdp2_ready(model: Any, optimizer: Any, accelerator: Any) -> None:
    """Post-prepare invariants: real shards, no meta, optimizer on prepared params."""
    import torch
    from torch.distributed.fsdp import FSDPModule

    if not getattr(accelerator, "is_fsdp2", False):
        raise RuntimeError("assert_fsdp2_ready requires the FSDP2 accelerator")
    decoder_names = set(getattr(model, "_trim_fsdp2_decoder_names", set()) or set())
    wrapped = []
    for name in sorted(decoder_names):
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        if isinstance(module, FSDPModule) or getattr(module, "_is_fsdp_managed_module", False):
            wrapped.append(name)
    if not wrapped:
        raise RuntimeError("no decoder layer became an FSDPModule; wrap policy did not apply")
    meta = [name for name, param in model.named_parameters() if getattr(param.device, "type", "") == "meta"]
    if meta:
        raise RuntimeError(f"parameters left on meta after prepare: {meta[:8]}")
    frozen_dtensor = 0
    frozen_total = 0
    from torch.distributed.tensor import DTensor

    for _name, param in model.named_parameters():
        if param.requires_grad:
            continue
        frozen_total += 1
        if isinstance(param, DTensor):
            frozen_dtensor += 1
    if frozen_total and frozen_dtensor * 4 < frozen_total:
        raise RuntimeError(
            f"frozen base is not sharded ({frozen_dtensor}/{frozen_total} DTensor); refusing a replicated base"
        )
    identity = {id(param) for param in model.parameters()}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) not in identity:
                raise RuntimeError("optimizer still references pre-prepare parameter objects")
    for name, module in model.named_modules():
        if "Expert" in type(module).__name__ and not module.training:
            raise RuntimeError(f"expert module {name} is frozen in eval mode; model.train() is required")
    if not model.training:
        raise RuntimeError("prepared model is not in training mode")


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
    # Bound forward origin per module class: catches site-packages patches or
    # runtime kernel swaps that KERNELS.json used to miss (only base class file).
    bound_forwards: dict[str, dict[str, str]] = {}
    for module in model.modules():
        key = type(module).__name__
        if ("Attn" not in key and "Attention" not in key and "Expert" not in key and "Moe" not in key) or key in bound_forwards:
            continue
        forward = getattr(type(module), "forward", None)
        try:
            path = inspect.getfile(forward) if forward is not None else ""
        except Exception:
            path = ""
        entry = {"file": str(path), "sha256": ""}
        if path and Path(path).is_file():
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        bound_forwards[key] = entry
    trainable = [
        {
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "bytes": int(param.numel() * param.element_size()),
        }
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    training_flags = sorted(
        {type(module).__name__: bool(module.training) for module in model.modules() if "Expert" in type(module).__name__}.items()
    )
    software: dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            module = __import__(name)
            software[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            software[name] = "missing"
    return {
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "attention_modules": [name for name in names if "Attn" in name or "Attention" in name],
        "expert_modules": [name for name in names if "Expert" in name or "Moe" in name or "MoE" in name],
        "quantized": quant is not None,
        "dtypes": dtypes,
        "implementation_file": source,
        "implementation_sha256": source_sha,
        "bound_forwards": bound_forwards,
        "expert_training_flags": dict(training_flags),
        "trainable": trainable,
        "software": software,
    }


def trainable_checksum(model: Any) -> str:
    """Digest of the trainable tensors.

    Under FSDP2 every rank holds a different local shard, so DTensors are
    aggregated with ``full_tensor()`` first. Every rank must call this in the
    same module order: the aggregation is a collective.
    """
    import torch

    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tensor = param.detach()
        if type(tensor).__name__ == "DTensor":
            tensor = tensor.full_tensor()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.float().cpu().numpy().tobytes())
        del tensor
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
