"""Training / eval budget accounting for fair comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainBudget:
    n_samples: int
    max_steps: int | None
    epochs: float
    effective_batch_size: int
    lora_rank: int
    max_tokens: int
    seed: int

    @property
    def optimizer_steps_proxy(self) -> int:
        if self.max_steps is not None:
            return int(self.max_steps)
        steps = int(max(1, round(self.n_samples * self.epochs / max(self.effective_batch_size, 1))))
        return steps

    @property
    def token_flops_proxy(self) -> float:
        # Relative proxy only — not absolute FLOPs.
        return float(self.optimizer_steps_proxy * self.effective_batch_size * self.max_tokens * self.lora_rank)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["optimizer_steps_proxy"] = self.optimizer_steps_proxy
        d["token_flops_proxy"] = self.token_flops_proxy
        return d


def assert_budget_match(a: TrainBudget, b: TrainBudget, *, mode: str = "sample") -> None:
    if mode == "sample":
        if a.n_samples != b.n_samples:
            raise ValueError(f"sample budget mismatch: {a.n_samples} vs {b.n_samples}")
        if a.optimizer_steps_proxy != b.optimizer_steps_proxy:
            raise ValueError(
                f"step budget mismatch: {a.optimizer_steps_proxy} vs {b.optimizer_steps_proxy}"
            )
    elif mode == "flops":
        # Allow 5% relative slack for token-length variance.
        pa, pb = a.token_flops_proxy, b.token_flops_proxy
        if abs(pa - pb) / max(pa, pb, 1.0) > 0.05:
            raise ValueError(f"FLOPs proxy mismatch: {pa} vs {pb}")
    else:
        raise ValueError(f"unknown budget mode: {mode}")
