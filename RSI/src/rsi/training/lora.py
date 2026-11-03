"""Minimal cumulative LoRA (adapter_mode=cumulative).

The main path keeps one cumulative adapter: later rounds continue training on
the same base + promoted adapter — never a fresh randomly-initialized adapter
per round.  Merge is allowed only with an export/reload logprob equivalence
check and a full-weight identity update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from rsi.contracts.common import digest_of


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_suffixes: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # hybrid linear-attention blocks (Qwen3.x): the attention-equivalent
        # projections live under linear_attn.*
        "out_proj",
        "in_proj_qkv",
    )

    @property
    def scale(self) -> float:
        return self.alpha / max(1, self.rank)


def freeze_non_lora_(model: Any) -> int:
    """Freeze every parameter except LoRA A/B.  Returns trainable count.

    Cumulative-adapter main path (task book §9.5): training updates the LoRA
    adapter only; base weights stay frozen until an explicit merge.
    """
    trainable = 0
    lora_ids = set()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_ids.add(id(module.lora_A))
            lora_ids.add(id(module.lora_B))
    for p in model.parameters():
        if id(p) in lora_ids:
            p.requires_grad = True
            trainable += 1
        else:
            p.requires_grad = False
    return trainable


def lora_param_count(model: Any) -> int:
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            n += module.lora_A.numel() + module.lora_B.numel()
    return n


class LoRALinear(nn.Module):
    """Wraps a torch.nn.Linear with a low-rank residual (registered module).

    A/B live in float32 on the base module's device (optimizer-friendly);
    ``forward`` casts for the low-rank matmul so bf16/fp16 bases work — the
    original fp32-only matmul broke every GPU bf16 run at the first generate.
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        self.base = base
        self.cfg = cfg
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(base.in_features, cfg.rank, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(cfg.rank, base.out_features, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Any) -> Any:
        out = self.base(x)
        a = self.lora_A.to(dtype=x.dtype, device=x.device)
        b = self.lora_B.to(dtype=x.dtype, device=x.device)
        lora = (x @ a @ b) * self.cfg.scale
        return out + lora.to(out.dtype)


def apply_lora(model: Any, cfg: LoRAConfig) -> dict[str, LoRALinear]:
    """Replace matching Linear modules with LoRALinear; returns the wrappers."""
    replaced: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in cfg.target_suffixes):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapper = LoRALinear(module, cfg)
            setattr(parent, child, wrapper)
            replaced[name] = wrapper
    return replaced


def lora_state_dict(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            out[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return out


def load_lora_state_dict(model: Any, state: Mapping[str, Any]) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a, key_b = f"{name}.lora_A", f"{name}.lora_B"
            if key_a in state:
                with __import__("torch").no_grad():
                    module.lora_A.copy_(state[key_a])
            if key_b in state:
                with __import__("torch").no_grad():
                    module.lora_B.copy_(state[key_b])


def adapter_digest(state: Mapping[str, Any]) -> str:
    import torch

    payload = {
        k: [round(float(x), 6) for x in v.flatten().tolist()[:64]]  # shape-stable digest head
        for k, v in sorted(state.items())
        if hasattr(v, "flatten")
    }
    return digest_of(payload)


def merge_lora_(model: Any) -> None:
    """Fold LoRA weights into the base Linear modules (destructive).

    Callers must run an export/reload logprob equivalence check afterwards.
    """
    import torch

    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_A @ module.lora_B) * module.cfg.scale
                module.base.weight += delta.t().to(module.base.weight.dtype)
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.base)
