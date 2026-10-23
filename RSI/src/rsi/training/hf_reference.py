"""HF reference training backend (single process, small models, CPU/GPU).

Real gradients through a real causal LM.  Semantics (task book §9.1):

- CISPO rows: ``weight = stopgrad(clamp(exp(new_lp - old_lp), lo, hi))``,
  token term ``-weight * adv * new_lp * mask``, reduced by the global
  effective-action-token normalizer.
- OPD rows: gated projected-gap
  ``L = sum(w m g (sg[lT] - lS)) / sum(w m)`` with Teacher/gate stop-grad,
  scaled by ``lambda_opd``.  Weighted-sum + world-size scaling form lives in
  ``opd.losses.scale_projected_gap_loss`` for distributed backends.

The objective is explicitly the gated gap — never silently a PPO surrogate or
GRPO default.  ``test_loss_and_grad_parity`` pins the numerics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from rsi.contracts.errors import NotImplementedCapability
from rsi.opd.losses import (
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_OPD_GATE_BETA,
    cispo_token_loss,
    gated_projected_gap_loss,
)
from rsi.opd.records import OPDRow, RLRow
from rsi.training.backend import GlobalNormalizers


@dataclass
class HFReferenceBackend:
    model: Any
    lambda_opd: float = 0.01
    gate_beta: float = DEFAULT_OPD_GATE_BETA
    gate_mode: str = "sigmoid"
    clip_low: float = DEFAULT_CLIP_LOW
    clip_high: float = DEFAULT_CLIP_HIGH
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    name: str = "hf_reference"
    consumes_opd: bool = True
    device: str = "cpu"
    _optimizer: Any = field(default=None, repr=False)
    step_count: int = 0

    def __post_init__(self) -> None:
        from rsi.providers.tiny_hf import tune_torch_threads

        tune_torch_threads()
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.model.to(self.device)
        self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # -- logprob helper ----------------------------------------------------
    def _token_logprobs(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> torch.Tensor:
        """Differentiable logprobs of ``target_ids`` given ``prompt_ids``."""
        if len(target_ids) == 0:
            return torch.zeros(0, device=self.device, requires_grad=True)
        full = torch.tensor([list(prompt_ids) + list(target_ids)], dtype=torch.long, device=self.device)
        out = self.model(full)
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        offset = len(prompt_ids)
        gathered = []
        for i, tid in enumerate(target_ids):
            pos = offset + i - 1
            if pos < 0:
                gathered.append(torch.zeros((), device=self.device))
            else:
                gathered.append(logp[pos, tid])
        return torch.stack(gathered)

    def forward_logprobs(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            return [float(x) for x in self._token_logprobs(prompt_ids, target_ids).tolist()]

    # -- TrainingBackend ---------------------------------------------------
    def zero_grad(self) -> None:
        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)

    def backward_rl(self, rows: Sequence[RLRow], normalizers: GlobalNormalizers) -> dict[str, float]:
        if not rows:
            return {"rl_loss": 0.0, "n_rl_rows": 0.0}
        total = torch.zeros((), device=self.device)
        clip_hits = 0
        tokens = 0
        for row in rows:
            new_lp = self._token_logprobs(row.prompt_token_ids, row.action_token_ids)
            old_lp = torch.tensor(row.old_logprobs, dtype=torch.float32, device=self.device)
            mask = torch.tensor(row.action_mask, dtype=torch.float32, device=self.device)
            loss_tok = cispo_token_loss(
                new_lp,
                old_lp,
                row.advantage,
                mask,
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                normalizer=float(normalizers.global_action_tokens),
            )
            total = total + loss_tok
            with torch.no_grad():
                ratio = (new_lp.detach() - old_lp).exp()
                clipped = ratio.clamp(min=self.clip_low, max=self.clip_high)
                clip_hits += int((ratio != clipped).sum().item())
                tokens += int(mask.sum().item())
        total.backward()
        return {
            "rl_loss": float(total.detach().item()),
            "n_rl_rows": float(len(rows)),
            "ratio_clip_fraction": clip_hits / max(1, tokens),
        }

    def backward_opd(
        self, rows: Sequence[OPDRow], normalizers: GlobalNormalizers, *, lambda_opd: float | None = None
    ) -> dict[str, float]:
        lam = self.lambda_opd if lambda_opd is None else float(lambda_opd)
        if not rows:
            return {"opd_loss": 0.0, "n_opd_rows": 0.0}
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
            gap = gated_projected_gap_loss(
                student_lp,
                teacher_lp,
                mask,
                weight,
                gate_beta=self.gate_beta,
                gate_mode=self.gate_mode,
            )
            total = total + lam * gap
            n_tokens += int(mask.sum().item())
        if n_tokens == 0:
            return {"opd_loss": 0.0, "n_opd_rows": float(len(rows)), "skipped_no_tokens": 1.0}
        total.backward()
        grad_norm = self._grad_norm()
        return {
            "opd_loss": float(total.detach().item()),
            "n_opd_rows": float(len(rows)),
            "n_opd_tokens": float(n_tokens),
            "opd_grad_norm": grad_norm,
        }

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().float().norm(2).item() ** 2)
        return total ** 0.5

    def optimizer_step(self) -> dict[str, float]:
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self._optimizer.step()
        self.step_count += 1
        return {"grad_norm": float(norm), "optimizer_steps": float(self.step_count)}

    def export_policy(self) -> Any:
        from rsi.providers.tiny_hf import HFPolicy

        policy = HFPolicy(self.model, base_model_id=f"hf_reference-step{self.step_count}", device=self.device)
        policy.policy_artifact = policy.policy_artifact.__class__.make(
            policy.policy_artifact.base_model_id,
            kind="trained",
            parent_policy_id=policy.policy_artifact.policy_id,
            extra={"steps": self.step_count},
        )
        return policy
