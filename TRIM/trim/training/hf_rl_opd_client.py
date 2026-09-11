"""HF debug client for joint CISPO + SR-OPD.

backend=hf_debug. Two backward passes accumulate on the same graph;
optim_step applies them once. Does not rewrite Tinker CISPO.

Per optimizer step the caller should pass a sampled query group (see
``sample_groups_for_step``). Inside a step, CISPO/OPD run length-bucketed
micro-batches instead of one 20B forward/backward per datum.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch

from trim.training.hf_rl_batch import (
    HF_DEFAULT_HEARTBEAT_EVERY,
    HF_DEFAULT_MICRO_BATCH,
    HF_MAX_FULL_TOKENS,
    iter_length_microbatches,
    log_train,
    sample_groups_for_step,
    truncate_teacher_forced_pair,
)
from trim.training.tinker_opd_datum import TinkerOPDDatum

__all__ = [
    "HFDebugTrainingClient",
    "episode_relative_advantages",
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
        clip_low: float = 0.0,
        clip_high: float = 5.0,
        micro_batch_size: int = HF_DEFAULT_MICRO_BATCH,
        heartbeat_every: int = HF_DEFAULT_HEARTBEAT_EVERY,
        max_full_tokens: int = HF_MAX_FULL_TOKENS,
    ) -> None:
        self.backend = backend
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.heartbeat_every = max(1, int(heartbeat_every))
        self.max_full_tokens = int(max_full_tokens)
        self.calls: list[tuple] = []
        self._accumulating = False
        self._step_tag = 0

    def _ensure_accum(self) -> None:
        if not self._accumulating:
            self.backend.optimizer.zero_grad(set_to_none=True)
            self._accumulating = True

    def _align_context(
        self, prompt_ids: list[int], action_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        return truncate_teacher_forced_pair(
            prompt_ids, action_ids, max_full=self.max_full_tokens
        )

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

    def _prepare_cispo_row(self, row: Any) -> dict[str, Any] | None:
        prompt_ids = list(
            row.get("effective_prompt_ids")
            or row.get("prompt_ids")
            or self.backend.encode(row["prompt"])
        )
        action_ids = list(row.get("action_ids") or self.backend.encode(row["action_text"]))
        if not action_ids:
            return None
        token_logprobs = row.get("token_logprobs")
        action_mask = list(row.get("action_mask") or [1] * len(action_ids))
        if token_logprobs is None:
            raise ValueError(f"CISPO row missing token_logprobs for query={row.get('query_id')}")
        if len(token_logprobs) != len(action_ids):
            raise ValueError(
                f"CISPO token_logprobs length {len(token_logprobs)} != action_ids {len(action_ids)}"
            )
        if len(action_mask) != len(action_ids):
            raise ValueError(
                f"CISPO action_mask length {len(action_mask)} != action_ids {len(action_ids)}"
            )
        prompt_ids, action_ids = self._align_context(prompt_ids, action_ids)
        if len(action_ids) != len(token_logprobs):
            raise ValueError("context alignment changed action length without matching logprobs")
        return {
            "prompt_ids": prompt_ids,
            "action_ids": action_ids,
            "token_logprobs": [float(x) for x in token_logprobs],
            "action_mask": action_mask,
            "advantage": float(row.get("advantage") or 0.0),
        }

    def _cispo_backward(
        self, rows: Sequence[Any], *, loss_fn_config: dict[str, Any] | None = None
    ) -> dict[str, float]:
        cfg = dict(loss_fn_config or {})
        clip_low = float(cfg.get("clip_low_threshold", self.clip_low))
        clip_high = float(cfg.get("clip_high_threshold", self.clip_high))
        device = self.backend._device
        prepared: list[dict[str, Any]] = []
        for row in rows:
            item = self._prepare_cispo_row(row)
            if item is not None:
                prepared.append(item)
        if not prepared:
            return {"loss": 0.0, "n_datums": 0, "n_microbatches": 0, "micro_batch_size": self.micro_batch_size}
        global_tokens = sum(
            sum(1 for m in row["action_mask"] if m) for row in prepared
        )
        global_tokens = max(1, global_tokens)
        loss_sum = 0.0
        n = 0
        n_mb = 0
        t0 = time.perf_counter()
        t_fwd = 0.0
        ratio_clip_hits = 0
        ratio_total = 0
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
            chunk_losses = []
            for row, new_lp in zip(chunk, logps):
                if new_lp.numel() == 0:
                    continue
                if new_lp.numel() != len(row["action_ids"]):
                    raise ValueError("new logprob length mismatch")
                new_lp = new_lp.float()
                old_lp = torch.tensor(row["token_logprobs"], device=device, dtype=torch.float32)
                mask = torch.tensor(row["action_mask"], device=device, dtype=torch.float32)
                if not torch.isfinite(new_lp).all() or not torch.isfinite(old_lp).all():
                    raise ValueError("non-finite logprobs in CISPO row")
                ratio = (new_lp - old_lp).exp()
                weight = ratio.clamp(min=clip_low, max=clip_high).detach()
                ratio_clip_hits += int((ratio != weight).sum().item())
                ratio_total += int(mask.sum().item())
                adv = float(row["advantage"])
                per_tok = -(weight * adv * new_lp * mask)
                chunk_losses.append(per_tok.sum() / global_tokens)
                n += 1
            if chunk_losses:
                loss = torch.stack(chunk_losses).sum()
                if loss.requires_grad:
                    loss.backward()
                loss_sum += float(loss.detach().item())
                del loss
            del logps, chunk_losses
            self._heartbeat(
                phase="cispo",
                done=n,
                total=len(prepared),
                t0=t0,
                extra={"n_microbatches": n_mb, "student_fwd_s": round(t_fwd, 3)},
            )
        return {
            "loss": loss_sum,
            "n_datums": n,
            "n_microbatches": n_mb,
            "micro_batch_size": self.micro_batch_size,
            "student_fwd_s": round(t_fwd, 3),
            "n_effective_tokens": global_tokens,
            "ratio_clip_fraction": (ratio_clip_hits / max(1, ratio_total)),
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
            prompt_ids, resp_ids = self._align_context(prompt_ids, resp_ids)
            if len(weights) != len(resp_ids):
                raise ValueError("OPD CE weights length mismatch")
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
                w = torch.tensor(weights[: len(logp)], device=device, dtype=torch.float32)
                if w.numel() != logp.numel():
                    raise ValueError("OPD CE weight/logprob length mismatch")
                losses.append(-(logp.float() * w).sum())
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
            weights = list(raw.weights[n_p:]) if len(raw.weights) > n_p else [1.0] * len(resp_ids)
            meta["_supervision_weights"] = weights
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
        meta["_supervision_weights"] = list(raw.get("weights") or [1.0] * len(resp_ids))
        return prompt_ids, resp_ids, teacher_ids, meta

    def _opd_sampled_gap(self, datums: Sequence[Any]) -> dict[str, float]:
        """SEED: λ × token-mean[g · (sg[ℓ^T] − ℓ^S)] on CISPO sampled tokens."""
        from trim.training.sr_opd_loss import gated_sampled_gap_per_token

        prepared: list[tuple[list[int], list[int], list[int], list[float], float, float]] = []
        n_total = 0
        for raw in datums:
            prompt_ids, resp_ids, teacher_ids, meta = self._unpack_opd_row(raw)
            if not resp_ids:
                continue
            weights = list(meta.get("_supervision_weights") or [1.0] * len(resp_ids))
            if len(weights) != len(resp_ids):
                raise ValueError("OPD gap supervision weights length mismatch")
            prompt_ids, resp_ids = self._align_context(prompt_ids, resp_ids)
            if teacher_ids:
                teacher_ids, _resp_t = self._align_context(teacher_ids, resp_ids)
            else:
                teacher_ids = list(prompt_ids)
            if len(resp_ids) != len(weights):
                raise ValueError("OPD gap post-alignment length mismatch")
            lam = float(meta.get("lambda_opd") if meta.get("lambda_opd") is not None else 0.01)
            beta = float(meta.get("gate_beta") if meta.get("gate_beta") is not None else 5.0)
            prepared.append((prompt_ids, resp_ids, teacher_ids, weights, lam, beta))
            n_total += sum(1 for w in weights if w)
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
            student_lps = self._logprobs_many([(p, r) for p, r, _t, _w, _lam, _b in chunk], require_grad=True)
            t_student += time.perf_counter() - t_s0
            t_t0 = time.perf_counter()
            teacher_lps = self._logprobs_many([(t, r) for _p, r, t, _w, _lam, _b in chunk], require_grad=False)
            t_teacher += time.perf_counter() - t_t0
            losses = []
            for (_p, _r, _tid, weights, lam, beta), student_lp, teacher_lp in zip(
                chunk, student_lps, teacher_lps
            ):
                if student_lp.numel() == 0:
                    continue
                if student_lp.numel() != teacher_lp.numel():
                    raise ValueError("teacher/student target length mismatch in OPD gap")
                w = torch.tensor(weights[: len(student_lp)], device=student_lp.device, dtype=torch.float32)
                gap = gated_sampled_gap_per_token(student_lp.float(), teacher_lp.float(), gate_beta=beta)
                losses.append((gap * w).sum() * (float(lam) / denom))
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
        self._ensure_accum()
        rows = list(data)
        t0 = time.perf_counter()
        if loss_fn == "cross_entropy":
            payload = self._opd_loss(rows)
        elif loss_fn in {"sampled_gap", "reverse_kl"}:
            payload = self._opd_sampled_gap(rows)
        else:
            payload = self._cispo_backward(rows, loss_fn_config=loss_fn_config)
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
            n_effective_tokens=payload.get("n_effective_tokens"),
        )
        return payload

    async def optim_step_async(self, adam_params: Any) -> dict[str, float]:
        del adam_params
        t0 = time.perf_counter()
        for p in self.backend.model.parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError("non-finite gradient before optimizer step")
        self.backend.optimizer.step()
        self.backend.optimizer.zero_grad(set_to_none=True)
        self._accumulating = False
        self.calls.append(("opt",))
        elapsed = round(time.perf_counter() - t0, 3)
        log_train("hf_optim_step", step=self._step_tag, elapsed_s=elapsed)
        return {"ok": 1.0, "elapsed_s": elapsed}


def episode_relative_advantages(rl_rows: Sequence[dict[str, Any]]) -> list[float]:
    """Normalize terminal rewards per query group at episode granularity."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rl_rows:
        by_query.setdefault(str(row.get("query_id")), []).append(row)
    adv_by_row: dict[int, float] = {}
    for rows in by_query.values():
        episodes: dict[str, float] = {}
        for row in rows:
            ep = str(
                row.get("episode_id")
                or f"{row.get('query_id')}_r{row.get('rollout_idx', 0)}"
            )
            episodes[ep] = float(row.get("reward") or 0.0)
        vals = list(episodes.values())
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = math.sqrt(var)
        ep_adv = {
            ep: (0.0 if std < 1e-8 else (float(r) - mean) / std)
            for ep, r in episodes.items()
        }
        for row in rows:
            ep = str(
                row.get("episode_id")
                or f"{row.get('query_id')}_r{row.get('rollout_idx', 0)}"
            )
            adv_by_row[id(row)] = ep_adv[ep]
    return [adv_by_row[id(row)] for row in rl_rows]


def group_relative_advantages(rewards: list[float], group_ids: list[str]) -> list[float]:
    """Backward-compatible wrapper; prefer episode_relative_advantages for RL rows."""
    rows = [
        {"reward": r, "query_id": gid, "rollout_idx": i}
        for i, (r, gid) in enumerate(zip(rewards, group_ids))
    ]
    return episode_relative_advantages(rows)


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
