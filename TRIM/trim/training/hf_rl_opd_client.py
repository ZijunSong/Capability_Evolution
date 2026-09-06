"""HF debug client for joint CISPO + SR-OPD.

backend=hf_debug. Two backward passes accumulate on the same graph;
optim_step applies them once. Does not rewrite Tinker CISPO.

Per optimizer step the caller should pass a sampled query group (see
``sample_groups_for_step``). Inside a step, CISPO/OPD run length-bucketed
micro-batches instead of one 20B forward/backward per datum.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from trim.training.hf_rl_batch import (
    HF_DEFAULT_HEARTBEAT_EVERY,
    HF_DEFAULT_MICRO_BATCH,
    iter_length_microbatches,
    log_train,
    sample_groups_for_step,
)
from trim.training.tinker_opd_datum import TinkerOPDDatum

# Truncation shared with vLLM behavior-policy logprobs so CISPO ratios match.
CISPO_MAX_PROMPT_TOKENS = 384
CISPO_MAX_ACTION_TOKENS = 128

__all__ = [
    "CISPO_MAX_ACTION_TOKENS",
    "CISPO_MAX_PROMPT_TOKENS",
    "HFDebugTrainingClient",
    "group_relative_advantages",
    "restore_trainable",
    "sample_groups_for_step",
    "snapshot_trainable",
]


class HFDebugTrainingClient:
    """Duck-types the Tinker TrainingClient used by hybrid_train_substep."""

    backend_name = "hf_debug"

    def __init__(
        self,
        backend: Any,
        *,
        clip_high: float = 5.0,
        micro_batch_size: int = HF_DEFAULT_MICRO_BATCH,
        heartbeat_every: int = HF_DEFAULT_HEARTBEAT_EVERY,
    ) -> None:
        self.backend = backend
        self.clip_high = float(clip_high)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.heartbeat_every = max(1, int(heartbeat_every))
        self.calls: list[tuple] = []
        self._accumulating = False
        self._step_tag = 0

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

    def _logprobs_many(
        self,
        pairs: list[tuple[list[int], list[int]]],
        *,
        require_grad: bool,
    ) -> list[torch.Tensor]:
        if not pairs:
            return []
        batch_fn = getattr(self.backend, "_teacher_forced_logprobs_batch", None)
        oom_exc: BaseException | None = None
        try:
            if callable(batch_fn) and len(pairs) > 1:
                return list(batch_fn(pairs, require_grad=require_grad))
        except torch.cuda.OutOfMemoryError as exc:
            oom_exc = exc
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            oom_exc = exc
        if oom_exc is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(pairs) == 1:
                raise oom_exc
            mid = max(1, len(pairs) // 2)
            return self._logprobs_many(pairs[:mid], require_grad=require_grad) + self._logprobs_many(
                pairs[mid:], require_grad=require_grad
            )
        return [
            self.backend._teacher_forced_logprobs(prompt, resp, require_grad=require_grad)
            for prompt, resp in pairs
        ]

    def _heartbeat(self, *, phase: str, done: int, total: int, t0: float, extra: dict[str, Any] | None = None) -> None:
        every = self.heartbeat_every
        n_mb = int((extra or {}).get("n_microbatches") or 0)
        if n_mb not in {1, 0} and done not in {1, total} and done % every != 0:
            return
        payload = {
            "phase": phase,
            "step": self._step_tag,
            "done": int(done),
            "total": int(total),
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "micro_batch_size": self.micro_batch_size,
        }
        if extra:
            payload.update(extra)
        log_train("hf_fb", **payload)

    def _cispo_backward(self, rows: Sequence[Any]) -> dict[str, float]:
        device = self.backend._device
        prepared: list[dict[str, Any]] = []
        for row in rows:
            prompt_ids = list(row.get("prompt_ids") or self.backend.encode(row["prompt"]))
            action_ids = list(row.get("action_ids") or self.backend.encode(row["action_text"]))
            if not action_ids:
                continue
            prompt_ids, action_ids = self._truncate_pair(prompt_ids, action_ids)
            prepared.append(
                {
                    "prompt_ids": prompt_ids,
                    "action_ids": action_ids,
                    "logprob_old": row.get("logprob_old"),
                    "advantage": float(row.get("advantage") or 0.0),
                }
            )
        scale = max(1, len(rows))
        if not prepared:
            return {"loss": 0.0, "n_datums": 0, "n_microbatches": 0, "micro_batch_size": self.micro_batch_size}
        total = 0.0
        n = 0
        n_mb = 0
        t0 = time.perf_counter()
        t_fwd = 0.0
        for chunk in iter_length_microbatches(
            prepared,
            size=self.micro_batch_size,
            length_fn=lambda row: len(row["prompt_ids"]) + len(row["action_ids"]),
        ):
            n_mb += 1
            t_fwd0 = time.perf_counter()
            logps = self._logprobs_many(
                [(row["prompt_ids"], row["action_ids"]) for row in chunk],
                require_grad=True,
            )
            t_fwd += time.perf_counter() - t_fwd0
            losses = []
            for row, logp in zip(chunk, logps):
                if logp.numel() == 0:
                    continue
                old = row["logprob_old"]
                if old is None:
                    ratio = torch.ones((), device=device, dtype=logp.dtype)
                else:
                    old_t = torch.tensor(float(old), device=device, dtype=logp.dtype)
                    ratio = torch.exp((logp.detach().mean() - old_t).clamp(-20, 20))
                weight = ratio.clamp(0.0, self.clip_high)
                losses.append(-(weight * float(row["advantage"]) * logp.mean()) / scale)
                n += 1
            if losses:
                loss = torch.stack(losses).sum()
                if loss.requires_grad:
                    loss.backward()
                total += float(loss.detach().item()) * scale
                del loss
            del logps, losses
            self._heartbeat(
                phase="cispo",
                done=n,
                total=len(prepared),
                t0=t0,
                extra={"n_microbatches": n_mb, "student_fwd_s": round(t_fwd, 3)},
            )
        return {
            "loss": total / max(1, n),
            "n_datums": n,
            "n_microbatches": n_mb,
            "micro_batch_size": self.micro_batch_size,
            "student_fwd_s": round(t_fwd, 3),
        }

    def _opd_loss(self, datums: Sequence[Any]) -> dict[str, float]:
        """Accumulate SR-OPD CE gradients on length-bucketed micro-batches."""
        device = self.backend._device
        prepared: list[tuple[list[int], list[int], list[float]]] = []
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
            prepared.append((prompt_ids, resp_ids, weights))
        if not prepared:
            return {"loss": 0.0, "n_datums": 0, "n_microbatches": 0, "micro_batch_size": self.micro_batch_size}
        total = 0.0
        n = 0
        n_mb = 0
        t0 = time.perf_counter()
        t_fwd = 0.0
        for chunk in iter_length_microbatches(
            prepared,
            size=self.micro_batch_size,
            length_fn=lambda row: len(row[0]) + len(row[1]),
        ):
            n_mb += 1
            t_fwd0 = time.perf_counter()
            logps = self._logprobs_many([(p, r) for p, r, _w in chunk], require_grad=True)
            t_fwd += time.perf_counter() - t_fwd0
            losses = []
            for (_p, _r, weights), logp in zip(chunk, logps):
                if logp.numel() == 0:
                    continue
                w = torch.tensor(weights[: len(logp)], device=device, dtype=logp.dtype)
                if w.numel() != logp.numel():
                    w = torch.ones_like(logp)
                losses.append(-(logp * w).sum())
                n += 1
            if losses:
                loss = torch.stack(losses).sum()
                if loss.requires_grad:
                    loss.backward()
                total += float(loss.detach().item())
                del loss
            del logps, losses
            self._heartbeat(
                phase="opd_ce",
                done=n,
                total=len(prepared),
                t0=t0,
                extra={"n_microbatches": n_mb, "student_fwd_s": round(t_fwd, 3)},
            )
        return {
            "loss": total / max(1, n),
            "n_datums": n,
            "n_microbatches": n_mb,
            "micro_batch_size": self.micro_batch_size,
            "student_fwd_s": round(t_fwd, 3),
        }

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
        from trim.training.sr_opd_loss import gated_sampled_gap_per_token

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
            return {"loss": 0.0, "n_datums": 0, "n_microbatches": 0, "micro_batch_size": self.micro_batch_size}

        total = 0.0
        denom = float(n_total)
        n = 0
        n_mb = 0
        t0 = time.perf_counter()
        t_student = 0.0
        t_teacher = 0.0
        for chunk in iter_length_microbatches(
            prepared,
            size=self.micro_batch_size,
            length_fn=lambda row: len(row[0]) + len(row[1]),
        ):
            n_mb += 1
            t_s0 = time.perf_counter()
            student_lps = self._logprobs_many([(p, r) for p, r, _t, _lam, _b in chunk], require_grad=True)
            t_student += time.perf_counter() - t_s0
            t_t0 = time.perf_counter()
            teacher_lps = self._logprobs_many([(t, r) for _p, r, t, _lam, _b in chunk], require_grad=False)
            t_teacher += time.perf_counter() - t_t0
            losses = []
            for (_p, _r, _tid, lam, beta), student_lp, teacher_lp in zip(chunk, student_lps, teacher_lps):
                if student_lp.numel() == 0:
                    continue
                gap = gated_sampled_gap_per_token(student_lp, teacher_lp, gate_beta=beta)
                losses.append(gap.sum() * (float(lam) / denom))
                n += 1
            if losses:
                loss = torch.stack(losses).sum()
                if loss.requires_grad:
                    loss.backward()
                total += float(loss.detach().item())
                del loss
            del student_lps, teacher_lps, losses
            self._heartbeat(
                phase="opd_gap",
                done=n,
                total=len(prepared),
                t0=t0,
                extra={
                    "n_microbatches": n_mb,
                    "student_fwd_s": round(t_student, 3),
                    "teacher_fwd_s": round(t_teacher, 3),
                },
            )
        return {
            "loss": total,
            "n_datums": n,
            "n_microbatches": n_mb,
            "micro_batch_size": self.micro_batch_size,
            "student_fwd_s": round(t_student, 3),
            "teacher_fwd_s": round(t_teacher, 3),
        }

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
        t0 = time.perf_counter()
        if loss_fn == "cross_entropy":
            payload = self._opd_loss(rows)
        elif loss_fn in {"sampled_gap", "reverse_kl"}:
            payload = self._opd_sampled_gap(rows)
        else:
            payload = self._cispo_backward(rows)
        payload = dict(payload)
        payload["elapsed_s"] = round(time.perf_counter() - t0, 3)
        payload["loss_fn"] = str(loss_fn)
        self.calls.append(("fb", loss_fn, len(rows)))
        log_train(
            "hf_fb_done",
            phase=str(loss_fn),
            step=self._step_tag,
            n_datums=int(payload.get("n_datums") or len(rows)),
            n_submitted=len(rows),
            n_microbatches=int(payload.get("n_microbatches") or 0),
            micro_batch_size=self.micro_batch_size,
            elapsed_s=payload["elapsed_s"],
            student_fwd_s=payload.get("student_fwd_s"),
            teacher_fwd_s=payload.get("teacher_fwd_s"),
        )
        return payload

    async def optim_step_async(self, adam_params: Any) -> dict[str, float]:
        del adam_params
        t0 = time.perf_counter()
        self.backend.optimizer.step()
        self.backend.optimizer.zero_grad(set_to_none=True)
        self._accumulating = False
        self.calls.append(("opt",))
        elapsed = round(time.perf_counter() - t0, 3)
        log_train("hf_optim_step", step=self._step_tag, elapsed_s=elapsed)
        return {"ok": 1.0, "elapsed_s": elapsed}


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
