"""Production training backend: torch FSDP2 (``torch_fsdp2``).

Model sharding, PEFT/LoRA support, mixed precision, checkpointing and export
must be profiled on the target hardware — prior OOM experience does not
transfer to new versions.  This backend requires an initialized
``torch.distributed`` process group (torchrun); without one it fails loudly
(``test_unimplemented_fails``) instead of pretending to train.

Distributed normalization uses ONE global effective-token/weight statistic;
local numerators follow the documented world-size scaling convention
(``opd.losses.scale_projected_gap_loss``) and a global denominator is never
all-reduced twice.  Rollout and training workers may occupy GPUs in phases,
handing off via immutable PolicyArtifacts.  First single-host, then torchrun
multi-host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from rsi.contracts.errors import NotImplementedCapability
from rsi.opd.losses import (
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_OPD_GATE_BETA,
    cispo_token_loss,
    gated_projected_gap_loss,
    scale_projected_gap_loss,
)
from rsi.opd.records import OPDRow, RLRow
from rsi.training.backend import GlobalNormalizers


def _require_dist() -> Any:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise NotImplementedCapability(
            "torch_fsdp2 backend requires an initialized torch.distributed "
            "process group (launch with torchrun). Refusing fake single-process "
            "fallback — use backend=hf_reference for single-process runs."
        )
    return dist


@dataclass
class FSDP2Backend:
    model: Any
    lambda_opd: float = 0.01
    gate_beta: float = DEFAULT_OPD_GATE_BETA
    gate_mode: str = "sigmoid"
    clip_low: float = DEFAULT_CLIP_LOW
    clip_high: float = DEFAULT_CLIP_HIGH
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    name: str = "torch_fsdp2"
    consumes_opd: bool = True
    _optimizer: Any = field(default=None, repr=False)
    step_count: int = 0

    def __post_init__(self) -> None:
        dist = _require_dist()
        import torch
        from torch.distributed.fsdp import fully_shard

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # FSDP2: shard transformer blocks + root; keep LoRA params unsharded-if-small.
        for block in getattr(self.model, "model", torch.nn.ModuleList()).children() if hasattr(self.model, "model") else []:
            fully_shard(block)
        fully_shard(self.model)
        self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def zero_grad(self) -> None:
        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)

    def _token_logprobs(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> Any:
        import torch

        if len(target_ids) == 0:
            return torch.zeros(0, device=self.device, requires_grad=True)
        full = torch.tensor([list(prompt_ids) + list(target_ids)], dtype=torch.long, device=self.device)
        out = self.model(full)
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        offset = len(prompt_ids)
        gathered = []
        for i, tid in enumerate(target_ids):
            pos = offset + i - 1
            gathered.append(logp[pos, tid] if pos >= 0 else torch.zeros((), device=self.device))
        return torch.stack(gathered)

    def backward_rl(self, rows: Sequence[RLRow], normalizers: GlobalNormalizers) -> dict[str, float]:
        import torch

        if not rows:
            return {"rl_loss": 0.0}
        total = torch.zeros((), device=self.device)
        for row in rows:
            new_lp = self._token_logprobs(row.prompt_token_ids, row.action_token_ids)
            old_lp = torch.tensor(row.old_logprobs, dtype=torch.float32, device=self.device)
            mask = torch.tensor(row.action_mask, dtype=torch.float32, device=self.device)
            total = total + cispo_token_loss(
                new_lp,
                old_lp,
                row.advantage,
                mask,
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                normalizer=float(normalizers.global_action_tokens),
            )
        total.backward()
        return {"rl_loss": float(total.detach().item()), "n_rl_rows": float(len(rows))}

    def backward_opd(
        self, rows: Sequence[OPDRow], normalizers: GlobalNormalizers, *, lambda_opd: float | None = None
    ) -> dict[str, float]:
        import torch

        lam = self.lambda_opd if lambda_opd is None else float(lambda_opd)
        if not rows:
            return {"opd_loss": 0.0, "skipped_no_tokens": 1.0}
        total = torch.zeros((), device=self.device)
        n_tokens = 0
        for row in rows:
            if not row.target_token_ids:
                continue
            student_lp = self._token_logprobs(row.student_prompt_token_ids, row.target_token_ids)
            teacher_lp = torch.tensor(row.teacher_logprobs, dtype=torch.float32, device=self.device)
            mask = torch.tensor(row.target_mask, dtype=torch.float32, device=self.device)
            weight = torch.tensor(
                row.token_weight or [1.0] * len(row.target_token_ids),
                dtype=torch.float32,
                device=self.device,
            )
            # Weighted-sum form with world-size scaling (global Z_gap, not
            # all-reduced again).
            from rsi.opd.losses import gated_action_gap_weighted_sum

            numer = gated_action_gap_weighted_sum(
                student_lp, teacher_lp, weight * mask, gate_beta=self.gate_beta, gate_mode=self.gate_mode
            )
            scaled = scale_projected_gap_loss(
                numer,
                lambda_opd=lam,
                z_gap=normalizers.global_opd_weight,
                world_size=normalizers.world_size,
            )
            total = total + scaled
            n_tokens += int(mask.sum().item())
        if n_tokens == 0:
            return {"opd_loss": 0.0, "skipped_no_tokens": 1.0}
        total.backward()
        return {"opd_loss": float(total.detach().item()), "n_opd_tokens": float(n_tokens)}

    def optimizer_step(self) -> dict[str, float]:
        import torch

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self._optimizer.step()
        self.step_count += 1
        return {"grad_norm": float(norm), "optimizer_steps": float(self.step_count), "rank": float(self.rank)}

    def export_policy(self) -> Any:
        raise NotImplementedCapability(
            "FSDP2 export uses the sharded checkpoint pipeline (DCP + LoRA "
            "aggregation); call training.checkpoint.save_checkpoint + "
            "training.export.export_adapter on the consolidated state"
        )
