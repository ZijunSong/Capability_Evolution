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

from rsi.contracts.common import digest_of


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_suffixes: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    @property
    def scale(self) -> float:
        return self.alpha / max(1, self.rank)


class LoRALinear:
    """Wraps a torch.nn.Linear with a low-rank residual."""

    def __init__(self, base: Any, cfg: LoRAConfig, torch_module: Any):
        torch = torch_module
        self.base = base
        self.cfg = cfg
        self.lora_A = torch.nn.Parameter(torch.zeros(base.in_features, cfg.rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(cfg.rank, base.out_features))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: Any) -> Any:
        torch = _torch_of(x)
        out = self.base(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.cfg.scale
        return out + lora.to(out.dtype)

    def parameters(self):
        return [self.lora_A, self.lora_B]


def _torch_of(x: Any):
    return __import__("torch")


def apply_lora(model: Any, cfg: LoRAConfig) -> dict[str, LoRALinear]:
    """Replace matching Linear modules with LoRALinear; returns the wrappers."""
    import torch

    replaced: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and any(name.endswith(s) for s in cfg.target_suffixes):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapper = LoRALinear(module, cfg, torch)
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
