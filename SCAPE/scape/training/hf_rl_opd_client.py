"""HF debug client for joint CISPO + SR-OPD.

backend=hf_debug. Two backward passes accumulate on the same graph;
optim_step applies them once. Does not rewrite Tinker CISPO.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from scape.training.tinker_opd_datum import TinkerOPDDatum

# Truncation shared with vLLM behavior-policy logprobs so CISPO ratios match.
CISPO_MAX_PROMPT_TOKENS = 384
CISPO_MAX_ACTION_TOKENS = 128


class HFDebugTrainingClient:
    """Duck-types the Tinker TrainingClient used by hybrid_train_substep."""

    backend_name = "hf_debug"

    def __init__(self, backend: Any, *, clip_high: float = 5.0) -> None:
        self.backend = backend
        self.clip_high = float(clip_high)
        self.calls: list[tuple] = []
        self._accumulating = False

    def _ensure_accum(self) -> None:
        if not self._accumulating:
            self.backend.optimizer.zero_grad(set_to_none=True)
            self._accumulating = True

    def _truncate_pair(self, prompt_ids: list[int], action_ids: list[int]) -> tuple[list[int], list[int]]:
        # Long Harmony prefixes + full logits OOM on gpt-oss MoE during backward.
        if len(prompt_ids) > CISPO_MAX_PROMPT_TOKENS:
            prompt_ids = prompt_ids[-CISPO_MAX_PROMPT_TOKENS:]
        if len(action_ids) > CISPO_MAX_ACTION_TOKENS:
            action_ids = action_ids[:CISPO_MAX_ACTION_TOKENS]
        return prompt_ids, action_ids

    def _cispo_backward(self, rows: Sequence[Any]) -> dict[str, float]:
        device = self.backend._device
        total = 0.0
        n = 0
        scale = max(1, len(rows))
        for row in rows:
            prompt_ids = list(row.get("prompt_ids") or self.backend.encode(row["prompt"]))
            action_ids = list(row.get("action_ids") or self.backend.encode(row["action_text"]))
            if not action_ids:
                continue
            prompt_ids, action_ids = self._truncate_pair(prompt_ids, action_ids)
            logp = self.backend._teacher_forced_logprobs(
                prompt_ids, action_ids, require_grad=True
            )
            old = row.get("logprob_old")
            if old is None:
                ratio = torch.ones((), device=device, dtype=logp.dtype)
            else:
                old_t = torch.tensor(float(old), device=device, dtype=logp.dtype)
                ratio = torch.exp((logp.detach().mean() - old_t).clamp(-20, 20))
            weight = ratio.clamp(0.0, self.clip_high)
            adv = float(row.get("advantage") or 0.0)
            loss = -(weight * adv * logp.mean()) / scale
            if loss.requires_grad:
                loss.backward()
            total += float(loss.detach().item()) * scale
            n += 1
            del logp, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"loss": total / max(1, n)}

    def _opd_loss(self, datums: Sequence[Any]) -> torch.Tensor:
        device = self.backend._device
        parts: list[torch.Tensor] = []
        for raw in datums:
            if isinstance(raw, TinkerOPDDatum):
                prompt_ids = list(raw.prompt_token_ids)
                n_p = len(prompt_ids)
                resp_ids = list(raw.target_tokens[n_p:])
                weights = list(raw.weights[n_p:])
            else:
                prompt_ids = list(raw.get("prompt_ids") or self.backend.encode(raw["prompt"]))
                resp_ids = list(raw.get("target_ids") or self.backend.encode(raw["target_text"]))
                weights = list(raw.get("weights") or [1.0] * len(resp_ids))
            if not resp_ids:
                continue
            prompt_ids, resp_ids = self._truncate_pair(prompt_ids, resp_ids)
            logp = self.backend._teacher_forced_logprobs(
                prompt_ids, resp_ids, require_grad=True
            )
            w = torch.tensor(weights[: len(logp)], device=device, dtype=logp.dtype)
            if w.numel() != logp.numel():
                w = torch.ones_like(logp)
            # Tinker-style: sum(w * nll) so pre-normalized weights already carry λ.
            parts.append(-(logp * w).sum())
        if not parts:
            return torch.zeros((), device=device, requires_grad=True)
        return torch.stack(parts).sum()

    async def forward_backward_async(
        self,
        data: Sequence[Any],
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del loss_fn_config
        self._ensure_accum()
        self.backend.model.train()
        rows = list(data)
        if loss_fn == "cross_entropy":
            loss = self._opd_loss(rows)
            if loss.requires_grad:
                loss.backward()
            payload = {"loss": float(loss.detach().item())}
        else:
            payload = self._cispo_backward(rows)
        self.calls.append(("fb", loss_fn, len(rows)))
        return payload

    async def optim_step_async(self, adam_params: Any) -> dict[str, float]:
        del adam_params
        self.backend.optimizer.step()
        self.backend.optimizer.zero_grad(set_to_none=True)
        self._accumulating = False
        self.calls.append(("opt",))
        return {"ok": 1.0}


def group_relative_advantages(rewards: list[float], group_ids: list[str]) -> list[float]:
    buckets: dict[str, list[int]] = {}
    for i, gid in enumerate(group_ids):
        buckets.setdefault(gid, []).append(i)
    out = [0.0] * len(rewards)
    for idxs in buckets.values():
        vals = [float(rewards[i]) for i in idxs]
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = var**0.5
        for i in idxs:
            out[i] = 0.0 if std < 1e-8 else (float(rewards[i]) - mean) / std
    return out


def snapshot_trainable(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable(model: Any, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snap:
                param.copy_(snap[name].to(param.device))
