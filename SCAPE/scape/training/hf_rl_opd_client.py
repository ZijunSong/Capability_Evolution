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

    def _opd_loss(self, datums: Sequence[Any]) -> dict[str, float]:
        """Accumulate SR-OPD gradients without retaining every datum graph.

        The datum weights already contain the global SR-OPD normalization and
        lambda, so summing per-datum backward passes is mathematically
        equivalent to one backward over the summed loss.  Keeping only one
        teacher-forced graph alive at a time is important for large MoE models.
        """
        device = self.backend._device
        total = 0.0
        n = 0
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
            row_loss = -(logp * w).sum()
            if row_loss.requires_grad:
                row_loss.backward()
            total += float(row_loss.detach().item())
            n += 1
            del logp, w, row_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"loss": total / max(1, n)}

    def _opd_reverse_kl(self, datums: Sequence[Any]) -> dict[str, float]:
        """Back-compat alias: scape+rl now uses the SEED sampled-gap contract."""
        return self._opd_sampled_gap(datums)

    def _unpack_opd_row(self, raw: Any) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
        if isinstance(raw, TinkerOPDDatum):
            prompt_ids = list(raw.prompt_token_ids)
            n_p = len(prompt_ids)
            resp_ids = list(raw.target_tokens[n_p:])
            teacher_ids = list(raw.teacher_prompt_token_ids or [])
            meta = dict(raw.metadata or {})
            meta.setdefault("lambda_opd", 0.01)
            meta.setdefault("gate_beta", 5.0)
            return prompt_ids, resp_ids, teacher_ids, meta
        prompt_ids = list(raw.get("prompt_ids") or self.backend.encode(raw["prompt"]))
        resp_ids = list(raw.get("target_ids") or self.backend.encode(raw["target_text"]))
        teacher_ids = list(raw.get("teacher_prompt_ids") or [])
        if not teacher_ids and raw.get("prompt_full"):
            teacher_ids = list(self.backend.encode(str(raw["prompt_full"])))
        meta = dict(raw.get("metadata") or {})
        if raw.get("lambda_opd") is not None:
            meta["lambda_opd"] = float(raw["lambda_opd"])
        if raw.get("gate_beta") is not None:
            meta["gate_beta"] = float(raw["gate_beta"])
        return prompt_ids, resp_ids, teacher_ids, meta

    def _opd_sampled_gap(self, datums: Sequence[Any]) -> dict[str, float]:
        """SEED: λ × token-mean[g · (sg[ℓ^T] − ℓ^S)] on CISPO sampled tokens.

        Student prefix is the same Harmony ids CISPO used when present.
        Teacher prefix is the privileged DualView sidecar. Gradients only
        through student logprobs of the sampled action.
        """
        from scape.training.sr_opd_loss import gated_sampled_gap_per_token

        prepared: list[tuple[list[int], list[int], list[int], float, float]] = []
        n_total = 0
        for raw in datums:
            prompt_ids, resp_ids, teacher_ids, meta = self._unpack_opd_row(raw)
            if not resp_ids:
                continue
            prompt_ids, resp_ids = self._truncate_pair(prompt_ids, resp_ids)
            if teacher_ids:
                teacher_ids, _resp_t = self._truncate_pair(teacher_ids, resp_ids)
            else:
                teacher_ids = list(prompt_ids)
            if not resp_ids:
                continue
            lam = float(meta.get("lambda_opd") if meta.get("lambda_opd") is not None else 0.01)
            beta = float(meta.get("gate_beta") if meta.get("gate_beta") is not None else 5.0)
            prepared.append((prompt_ids, resp_ids, teacher_ids, lam, beta))
            n_total += len(resp_ids)
        if not prepared or n_total <= 0:
            return {"loss": 0.0}

        total = 0.0
        denom = float(n_total)
        for prompt_ids, resp_ids, teacher_ids, lam, beta in prepared:
            student_lp = self.backend._teacher_forced_logprobs(
                prompt_ids, resp_ids, require_grad=True
            )
            teacher_lp = self.backend._teacher_forced_logprobs(
                teacher_ids, resp_ids, require_grad=False
            )
            gap = gated_sampled_gap_per_token(student_lp, teacher_lp, gate_beta=beta)
            row_loss = gap.sum() * (float(lam) / denom)
            if row_loss.requires_grad:
                row_loss.backward()
            total += float(row_loss.detach().item())
            del student_lp, teacher_lp, gap, row_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"loss": total}

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
            payload = self._opd_loss(rows)
        elif loss_fn in {"sampled_gap", "reverse_kl"}:
            payload = self._opd_sampled_gap(rows)
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
