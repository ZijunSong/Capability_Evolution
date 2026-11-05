"""Probe C1 micro-updates: frozen-Teacher projected-gap on fixed states.

Objective (§7.3) reuses the SEED gated gap math of ``sr_opd_loss`` — it is a
gated NLL, not a non-negative KL, and its value may be negative:

    g_j = σ(β · sg[ℓ^T_j − ℓ^S_j])
    L   = (1/Z) Σ_j w_j g_j (sg[ℓ^T_j] − ℓ^S_j),   Z = Σ_j w_j

``Z`` is the effective supervision weight sum *outside* the gate. ``Z == 0``
skips the update and reports NO_SUPPORT. ``loss_scale = 1`` by definition of
C1 — the formal joint λ must not be copied in. Gradient accumulation uses one
global Z per optimizer step; per-microbatch normalization is forbidden.

Each candidate starts from the exact same θ0 (weights, optimizer moments,
scheduler, RNG). The Teacher stays frozen at θ0 for the whole C1 run.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from trim.probes.probe_c_cache import CachedSupervisionRow
from trim.training.sr_opd_loss import (
    action_gap_gate,
    gated_action_gap_weighted_sum,
    scale_projected_gap_loss,
)

OBJECTIVE_FROZEN_PROJECTED_GAP = "frozen_projected_gap"
NO_SUPPORT = "NO_SUPPORT"


class TrainContractError(ValueError):
    pass


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------


def frozen_projected_gap_numerator(
    student_logprobs: Any,
    teacher_logprobs: Any,
    weights: Any,
    *,
    gate_beta: float = 5.0,
    gate_mode: str = "sigmoid",
) -> Any:
    """Σ_j w_j g_j (sg[ℓ^T_j] − ℓ^S_j) without the Z division."""
    return gated_action_gap_weighted_sum(
        student_logprobs,
        teacher_logprobs,
        weights,
        gate_beta=gate_beta,
        gate_mode=gate_mode,
    )


def effective_weight_sum(weights: Sequence[float], mask: Sequence[bool] | None = None) -> float:
    """Z outside the gate. Gate values must never be folded into Z."""
    if mask is None:
        return float(sum(float(w) for w in weights))
    return float(sum(float(w) for w, m in zip(weights, mask) if m))


def scale_c1_loss(
    numerator: Any,
    *,
    z_gap: float,
    loss_scale: float = 1.0,
    world_size: int = 1,
) -> Any:
    """world_size · loss_scale · numer / Z  (λ := loss_scale = 1 for C1)."""
    return scale_projected_gap_loss(
        numerator,
        lambda_opd=float(loss_scale),
        z_gap=float(z_gap),
        world_size=int(world_size),
    )


# ---------------------------------------------------------------------------
# token-budget batching
# ---------------------------------------------------------------------------


@dataclass
class MicroBatchPlan:
    """Rows grouped so each optimizer step consumes ≥ budget effective tokens."""

    batches: list[list[CachedSupervisionRow]]
    tokens_per_batch: list[int]
    leftover_tokens: int
    n_unique_tokens: int
    n_rows: int
    n_reused_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_batches": len(self.batches),
            "tokens_per_batch": list(self.tokens_per_batch),
            "leftover_tokens": int(self.leftover_tokens),
            "n_unique_tokens": int(self.n_unique_tokens),
            "n_rows": int(self.n_rows),
            "n_reused_rows": int(self.n_reused_rows),
            "note": "repeated sampling of the same state is counted as reuse, "
            "not as new data volume",
        }


def plan_micro_batches(
    rows: Sequence[CachedSupervisionRow],
    *,
    action_tokens_per_optimizer_step: int = 1024,
    repeat: int = 1,
) -> MicroBatchPlan:
    """Group rows into optimizer steps of ~1024 effective supervision tokens.

    ``repeat`` cycles the fixed state set (4/8/16 steps on few states). The
    leftover batch is kept explicit and rows re-sampled by ``repeat`` are
    reported as reuse.
    """
    if action_tokens_per_optimizer_step <= 0:
        raise TrainContractError("action_tokens_per_optimizer_step must be > 0")
    if repeat <= 0:
        raise TrainContractError("repeat must be > 0")

    def eff_tokens(row: CachedSupervisionRow) -> int:
        return sum(1 for w in row.target.weights if w > 0)

    batches: list[list[CachedSupervisionRow]] = []
    tokens_per_batch: list[int] = []
    current: list[CachedSupervisionRow] = []
    current_tokens = 0
    unique_tokens = sum(eff_tokens(r) for r in rows)
    for _rep in range(repeat):
        for row in rows:
            if not row.target.target_action_ids:
                continue  # no target at all; not a training row
            t = eff_tokens(row)
            current.append(row)
            current_tokens += t
            if current_tokens >= action_tokens_per_optimizer_step:
                batches.append(current)
                tokens_per_batch.append(current_tokens)
                current, current_tokens = [], 0
    leftover = current_tokens
    if current:
        batches.append(current)
        tokens_per_batch.append(current_tokens)
    total_planned = sum(tokens_per_batch)
    n_rows = len(rows) * repeat
    return MicroBatchPlan(
        batches=batches,
        tokens_per_batch=tokens_per_batch,
        leftover_tokens=leftover,
        n_unique_tokens=unique_tokens,
        n_rows=n_rows,
        n_reused_rows=max(0, n_rows - len(rows)),
    )


# ---------------------------------------------------------------------------
# student backend protocol
# ---------------------------------------------------------------------------


class StudentBackend(Protocol):
    """Score and update the Student. Teacher scoring lives behind the cache."""

    def token_logprobs(
        self, prompt_ids: Sequence[int], target_ids: Sequence[int]
    ) -> list[float]:
        """Current-Student token logp for the target action (detached)."""
        ...

    def train_microbatch(
        self,
        *,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
        teacher_logprobs: Sequence[float],
        weights: Sequence[float],
        z_gap: float,
        gate_beta: float,
        loss_scale: float,
    ) -> dict[str, Any]:
        """One microbatch forward/backward. Returns loss numerator + stats.

        Must backprop ``scale_c1_loss(numer, z_gap=z_gap, loss_scale=...)``
        where ``numer`` uses ``frozen_projected_gap_numerator``. The Teacher
        side must not receive gradients.
        """
        ...

    def optimizer_step(self) -> dict[str, Any]:
        """Apply accumulated grads with global-Z scaling already handled."""
        ...

    def trainable_checksum(self) -> str:
        ...

    def save_trainable(self) -> dict[str, Any]:
        """Trainable weights + optimizer/scheduler/RNG metadata."""
        ...

    def load_trainable(self, payload: Mapping[str, Any]) -> None:
        ...


# ---------------------------------------------------------------------------
# candidate reset (§7.5)
# ---------------------------------------------------------------------------


def _checksum_bytes(obj: Any) -> bytes:
    import hashlib

    h = hashlib.sha256()
    if hasattr(obj, "detach"):  # torch tensor
        t = obj.detach().to("cpu").contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        try:
            h.update(bytes(t.view("uint8")))  # type: ignore[arg-type]
        except Exception:
            h.update(t.numpy().tobytes())
        return h.digest()
    if isinstance(obj, (list, tuple)):
        for item in obj:
            h.update(_checksum_bytes(item))
        return h.digest()
    if isinstance(obj, Mapping):
        for key in sorted(obj):
            h.update(str(key).encode())
            h.update(_checksum_bytes(obj[key]))
        return h.digest()
    h.update(repr(obj).encode())
    return h.digest()


def checksum_payload(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_checksum_bytes(dict(payload))).hexdigest()


@dataclass
class CandidateResetManager:
    """Exact per-candidate restart from θ0 (weights/optim/RNG)."""

    backend: StudentBackend
    theta0_checksum: str = ""
    theta0_payload: dict[str, Any] = field(default_factory=dict)
    n_resets: int = 0

    def capture_theta0(self) -> str:
        self.theta0_payload = dict(self.backend.save_trainable())
        self.theta0_checksum = checksum_payload(self.theta0_payload)
        return self.theta0_checksum

    def reset_exact(self, seed: int) -> dict[str, Any]:
        if not self.theta0_payload:
            raise TrainContractError("capture_theta0() must run before reset_exact()")
        self.backend.load_trainable(self.theta0_payload)
        got = checksum_payload(self.backend.save_trainable())
        if got != self.theta0_checksum:
            raise TrainContractError(
                f"reset failed to restore theta0: want={self.theta0_checksum} got={got}"
            )
        self.n_resets += 1
        return {"seed": int(seed), "theta0_checksum": self.theta0_checksum}

    def verify_theta0(self) -> None:
        got = checksum_payload(self.backend.save_trainable())
        if got != self.theta0_checksum:
            raise TrainContractError(
                f"theta0 checksum drift before candidate: want={self.theta0_checksum} got={got}"
            )

    def verify_updated(self) -> None:
        got = checksum_payload(self.backend.save_trainable())
        if got == self.theta0_checksum:
            raise TrainContractError(
                "trainable checksum unchanged after updates (model did not move)"
            )

    # -- real checkpoint (guide §10.2: checksum is NOT a checkpoint) -----
    def save_checkpoint_atomic(self, out_dir: Path, *, tag: str) -> dict[str, Any]:
        """Atomically persist trainable weights + optimizer + RNG + model
        weights checksum. tmp write → verify reloadable → rename."""
        import torch

        payload = self.backend.save_trainable()
        weights_ck = checksum_payload({"model": payload.get("model")})
        ck = {
            "tag": str(tag),
            "model_weights_checksum": weights_ck,
            "trainable_checksum": self.theta0_checksum and checksum_payload(payload),
            "saved_at": time.strftime("%F %T"),
        }
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"checkpoint_{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"meta": ck, "payload": payload}, tmp)
        # Verify reloadable before commit (same wrapping as the pre hash).
        reloaded = torch.load(tmp, map_location="cpu", weights_only=False)
        if checksum_payload({"model": reloaded["payload"].get("model")}) != weights_ck:
            tmp.unlink(missing_ok=True)
            raise TrainContractError(f"checkpoint verify failed for {tag}")
        tmp.replace(path)
        return ck


# ---------------------------------------------------------------------------
# C1 trainer
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    loss: float
    n_tokens: int
    z_gap: float
    grad_norm: float | None
    seconds: float
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "n_tokens": self.n_tokens,
            "z_gap": self.z_gap,
            "grad_norm": self.grad_norm,
            "seconds": self.seconds,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "loss_note": "gated NLL; may be negative; not a non-negative KL",
        }


@dataclass
class TrainRunResult:
    component_id: str
    train_seed: int
    steps: list[StepRecord]
    checkpoints: dict[int, str]  # step -> trainable checksum after that step
    total_tokens: int
    theta0_checksum: str
    objective: str = OBJECTIVE_FROZEN_PROJECTED_GAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "train_seed": self.train_seed,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "checkpoints": {str(k): v for k, v in self.checkpoints.items()},
            "total_tokens": self.total_tokens,
            "theta0_checksum": self.theta0_checksum,
        }


@dataclass
class MicroUpdateSession:
    """Persistent sampler state across budget stages (guide §10.1).

    4→8→16 are ABSOLUTE optimizer steps on ONE data trajectory: the batch
    plan is built once, the cursor survives stage boundaries, so
    ``train_until(session, 4)`` + ``train_until(session, 8)`` consumes the
    same stream as a direct 8-step run (4+4==8, test-enforced).
    """

    optimizer_step: int = 0
    consumed_action_tokens: int = 0
    batches: list[list[CachedSupervisionRow]] = field(default_factory=list)
    batch_tokens: list[int] = field(default_factory=list)
    rng_state: Any = None
    contract_hash: str = ""
    n_unique_rows: int = 0


class MicroUpdateTrainer:
    """Fixed-Teacher projected-gap micro-updates under a token budget."""

    def __init__(
        self,
        backend: StudentBackend,
        *,
        gate_beta: float = 5.0,
        loss_scale: float = 1.0,
        action_tokens_per_optimizer_step: int = 1024,
        world_size: int = 1,
    ) -> None:
        self.backend = backend
        self.gate_beta = float(gate_beta)
        self.loss_scale = float(loss_scale)
        self.action_tokens_per_optimizer_step = int(action_tokens_per_optimizer_step)
        self.world_size = int(world_size)

    # -- persistent-session path (4+4==8 contract) -----------------------
    def open_session(
        self,
        rows: Sequence[CachedSupervisionRow],
        *,
        train_seed: int,
    ) -> MicroUpdateSession:
        """Build the batch plan ONCE with a seed-deterministic shuffle."""
        import random

        ordered = list(rows)
        rng = random.Random(int(train_seed))
        rng.shuffle(ordered)
        plan = plan_micro_batches(
            ordered,
            action_tokens_per_optimizer_step=self.action_tokens_per_optimizer_step,
            repeat=1,
        )
        session = MicroUpdateSession(
            batches=list(plan.batches),
            batch_tokens=list(plan.tokens_per_batch),
            contract_hash=checksum_payload(
                {"rows": [r.state.state_id for r in ordered], "tokens": plan.tokens_per_batch}
            ),
            n_unique_rows=len(rows),
        )
        return session

    def train_until(
        self,
        session: MicroUpdateSession,
        rows: Sequence[CachedSupervisionRow],
        *,
        target_absolute_step: int,
        component_id: str = "",
        train_seed: int = 0,
        checkpoint_steps: Sequence[int] = (),
        checkpoint_dir: Path | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> TrainRunResult:
        """Advance ``session`` to ``target_absolute_step`` on the SAME stream.

        Extends the plan by cycling the fixed state set when the stream is
        exhausted (reuse recorded per MicroBatchPlan semantics); absolute
        step numbering is authoritative — callers never pass relative steps.
        """
        import torch as _torch

        step_records: list[StepRecord] = []
        checkpoints: dict[int, str] = {}
        seg_tokens = 0
        theta0 = self.backend.trainable_checksum()
        want = int(target_absolute_step)
        ckpt_set = set(int(c) for c in checkpoint_steps)
        while session.optimizer_step < want:
            if not session.batches:
                raise TrainContractError("session batch stream empty; cannot advance")
            # Cycle the plan when exhausted (deterministic same-stream reuse).
            if session.optimizer_step >= len(session.batches):
                cycle = session.optimizer_step // max(1, len(session.batches))
                src_idx = session.optimizer_step % len(session.batches)
            else:
                src_idx = session.optimizer_step
            batch = session.batches[src_idx]
            n_tok = session.batch_tokens[src_idx]
            start = time.time()
            z_gap = effective_weight_sum(
                [w for row in batch for w in row.target.weights]
            )
            if z_gap <= 0.0:
                rec = StepRecord(
                    step=session.optimizer_step + 1,
                    loss=0.0, n_tokens=0, z_gap=0.0, grad_norm=None,
                    seconds=time.time() - start, skipped=True, skip_reason=NO_SUPPORT,
                )
                step_records.append(rec)
                if on_step:
                    on_step(rec)
                session.optimizer_step += 1
                continue
            for row in batch:
                self.backend.train_microbatch(
                    prompt_ids=row.state.effective_student_prompt_ids,
                    target_ids=row.target.target_action_ids,
                    teacher_logprobs=row.target.teacher_token_logprobs,
                    weights=row.target.weights,
                    z_gap=z_gap,
                    gate_beta=self.gate_beta,
                    loss_scale=self.loss_scale,
                )
            step_stats = self.backend.optimizer_step()
            grad_norm = step_stats.get("grad_norm")
            rec = StepRecord(
                step=session.optimizer_step + 1,
                loss=float(step_stats.get("loss", 0.0)),
                n_tokens=n_tok,
                z_gap=z_gap,
                grad_norm=None if grad_norm is None else float(grad_norm),
                seconds=time.time() - start,
            )
            step_records.append(rec)
            seg_tokens += n_tok
            session.consumed_action_tokens += n_tok
            session.optimizer_step += 1
            if on_step:
                on_step(rec)
            if rec.step in ckpt_set:
                if checkpoint_dir is not None:
                    ck = self.save_session_checkpoint(session, checkpoint_dir, tag=f"step{rec.step:04d}")
                    checkpoints[rec.step] = ck["model_weights_checksum"]
                else:
                    checkpoints[rec.step] = self.backend.trainable_checksum()
        return TrainRunResult(
            component_id=component_id,
            train_seed=train_seed,
            steps=step_records,
            checkpoints=checkpoints,
            total_tokens=seg_tokens,
            theta0_checksum=theta0,
        )

    def save_session_checkpoint(
        self, session: MicroUpdateSession, out_dir: Path, *, tag: str
    ) -> dict[str, Any]:
        """Session + weights atomic checkpoint (guide §10.2)."""
        import torch as _torch

        payload = self.backend.save_trainable()
        weights_ck = checksum_payload({"model": payload.get("model")})
        ck = {
            "tag": str(tag),
            "optimizer_step": session.optimizer_step,
            "consumed_action_tokens": session.consumed_action_tokens,
            "contract_hash": session.contract_hash,
            "model_weights_checksum": weights_ck,
            "saved_at": time.strftime("%F %T"),
        }
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"checkpoint_{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        _torch.save({"meta": ck, "payload": payload, "session": {
            "optimizer_step": session.optimizer_step,
            "consumed_action_tokens": session.consumed_action_tokens,
            "contract_hash": session.contract_hash,
        }}, tmp)
        reloaded = _torch.load(tmp, map_location="cpu", weights_only=False)
        if checksum_payload({"model": reloaded["payload"].get("model")}) != weights_ck:
            tmp.unlink(missing_ok=True)
            raise TrainContractError(f"session checkpoint verify failed for {tag}")
        tmp.replace(path)
        return ck

    def train_to_budget(
        self,
        rows: Sequence[CachedSupervisionRow],
        *,
        target_steps: int,
        component_id: str = "",
        train_seed: int = 0,
        checkpoint_steps: Sequence[int] = (),
        objective: str = OBJECTIVE_FROZEN_PROJECTED_GAP,
        start_step: int = 0,
        theta0_checksum: str = "",
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> TrainRunResult:
        """Run ``target_steps`` optimizer updates continuing the same trajectory.

        ``start_step`` labels the step numbering when the 4→8 budget stages
        share one optimizer trajectory (checkpoint keys stay absolute).
        """
        if objective != OBJECTIVE_FROZEN_PROJECTED_GAP:
            raise TrainContractError(
                f"unknown objective {objective!r}; C1 supports "
                f"{OBJECTIVE_FROZEN_PROJECTED_GAP!r} only (no silent CE switch)"
            )
        plan = plan_micro_batches(
            rows,
            action_tokens_per_optimizer_step=self.action_tokens_per_optimizer_step,
            repeat=max(1, math.ceil(target_steps / max(1, len(rows) or 1))),
        )
        # Trim/extend plan to exactly the requested optimizer steps (4 → 8 keeps
        # the same optimizer trajectory; leftover batch is kept explicit).
        batches = list(plan.batches)
        tokens = list(plan.tokens_per_batch)
        while len(batches) < target_steps:
            if not rows:
                raise TrainContractError("no supervision rows to train on")
            # Repeat pass over the same fixed states (recorded as reuse).
            extra = plan_micro_batches(
                rows,
                action_tokens_per_optimizer_step=self.action_tokens_per_optimizer_step,
                repeat=1,
            )
            if not extra.batches:
                raise TrainContractError(
                    "rows produce no trainable batches (empty targets); refusing "
                    "to fake optimizer steps"
                )
            batches.extend(extra.batches)
            tokens.extend(extra.tokens_per_batch)
        batches = batches[:target_steps]
        tokens = tokens[:target_steps]

        step_records: list[StepRecord] = []
        checkpoints: dict[int, str] = {}
        total_tokens = 0
        theta0 = theta0_checksum or self.backend.trainable_checksum()
        for step_idx, (batch, n_tok) in enumerate(
            zip(batches, tokens), start=start_step + 1
        ):
            start = time.time()
            # Global Z over the whole optimizer step (never per microbatch).
            z_gap = effective_weight_sum(
                [w for row in batch for w in row.target.weights]
            )
            if z_gap <= 0.0:
                rec = StepRecord(
                    step=step_idx,
                    loss=0.0,
                    n_tokens=0,
                    z_gap=0.0,
                    grad_norm=None,
                    seconds=time.time() - start,
                    skipped=True,
                    skip_reason=NO_SUPPORT,
                )
                step_records.append(rec)
                if on_step:
                    on_step(rec)
                continue
            numer_total = 0.0
            for row in batch:
                stats = self.backend.train_microbatch(
                    prompt_ids=row.state.effective_student_prompt_ids,
                    target_ids=row.target.target_action_ids,
                    teacher_logprobs=row.target.teacher_token_logprobs,
                    weights=row.target.weights,
                    z_gap=z_gap,
                    gate_beta=self.gate_beta,
                    loss_scale=self.loss_scale,
                )
                numer_total += float(stats.get("numerator", 0.0))
            step_stats = self.backend.optimizer_step()
            grad_norm = step_stats.get("grad_norm")
            loss = float(step_stats.get("loss", numer_total / z_gap * self.loss_scale))
            rec = StepRecord(
                step=step_idx,
                loss=loss,
                n_tokens=n_tok,
                z_gap=z_gap,
                grad_norm=None if grad_norm is None else float(grad_norm),
                seconds=time.time() - start,
            )
            step_records.append(rec)
            total_tokens += n_tok
            if on_step:
                on_step(rec)
            if step_idx in set(int(c) for c in checkpoint_steps):
                checkpoints[step_idx] = self.backend.trainable_checksum()
        return TrainRunResult(
            component_id=component_id,
            train_seed=train_seed,
            steps=step_records,
            checkpoints=checkpoints,
            total_tokens=total_tokens,
            theta0_checksum=theta0,
        )


# ---------------------------------------------------------------------------
# reference torch backend (real gradients; tests use a tiny module)
# ---------------------------------------------------------------------------


class TorchStudentBackend:
    """Student backend over any torch module + tokenizer-free id lists.

    This is the *probe* trainer, not the SFT update path: the backward is the
    gated projected-gap loss only. Pass a real HF causal LM module for
    production use (see ``probe_c_runner`` wiring).
    """

    def __init__(
        self,
        model: Any,
        *,
        optimizer: Any,
        scheduler: Any = None,
        grad_clip: float = 1.0,
        device: str = "cpu",
        world_size: int = 1,
        loss_scale: float = 1.0,
        gate_beta: float = 5.0,
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = float(grad_clip)
        self.device = device
        self.world_size = int(world_size)
        self.loss_scale = float(loss_scale)
        self.gate_beta = float(gate_beta)
        self._rng_state = torch.random.get_rng_state()
        self._accum_numer: list[Any] = []

    # -- scoring --------------------------------------------------------
    def token_logprobs(
        self, prompt_ids: Sequence[int], target_ids: Sequence[int]
    ) -> list[float]:
        torch = self.torch
        logits = self._forward_logits(prompt_ids, target_ids)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tgt = torch.tensor(list(target_ids), dtype=torch.long, device=logits.device)
        gathered = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return [float(x) for x in gathered.detach().cpu().reshape(-1).tolist()]

    def _forward_logits(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> Any:
        torch = self.torch
        ids = torch.tensor(
            [list(prompt_ids) + list(target_ids)], dtype=torch.long, device=self.device
        )
        out = self.model(ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out.logits
        # Predict target tokens: positions of prompt last .. target end
        n_prompt = len(prompt_ids)
        return logits[0, n_prompt - 1 : n_prompt - 1 + len(target_ids), :]

    # -- training -------------------------------------------------------
    def train_microbatch(
        self,
        *,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
        teacher_logprobs: Sequence[float],
        weights: Sequence[float],
        z_gap: float,
        gate_beta: float,
        loss_scale: float,
    ) -> dict[str, Any]:
        torch = self.torch
        self.model.train()
        logits = self._forward_logits(prompt_ids, target_ids)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tgt = torch.tensor(list(target_ids), dtype=torch.long, device=logits.device)
        student = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).reshape(-1)
        teacher = torch.tensor(
            list(teacher_logprobs), dtype=student.dtype, device=student.device
        )
        w = torch.tensor(list(weights), dtype=student.dtype, device=student.device)
        if teacher.numel() != student.numel():
            raise TrainContractError(
                f"target token length mismatch: student={student.numel()} "
                f"teacher={teacher.numel()}"
            )
        numer = frozen_projected_gap_numerator(
            student, teacher, w, gate_beta=gate_beta
        )
        loss = scale_c1_loss(
            numer, z_gap=z_gap, loss_scale=loss_scale, world_size=self.world_size
        )
        loss.backward()
        self._accum_numer.append(numer.detach())
        return {
            "numerator": float(numer.detach().item()),
            "loss": float(loss.detach().item()),
            "gate_mean": float(
                action_gap_gate(
                    (teacher - student).detach(), gate_beta=gate_beta
                )
                .mean()
                .item()
            ),
        }

    def optimizer_step(self) -> dict[str, Any]:
        torch = self.torch
        grad_norm = None
        if self.grad_clip > 0:
            params = [p for p in self.model.parameters() if p.grad is not None]
            if params:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip).item()
                )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        numer = 0.0
        if self._accum_numer:
            numer = float(sum(float(x.item()) for x in self._accum_numer))
            self._accum_numer.clear()
        return {"grad_norm": grad_norm, "loss": numer}

    # -- state / checksums ---------------------------------------------
    def trainable_checksum(self) -> str:
        return checksum_payload(self.save_trainable())

    def save_trainable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "rng_state": self.torch.random.get_rng_state(),
        }
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        return payload

    def load_trainable(self, payload: Mapping[str, Any]) -> None:
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "rng_state" in payload:
            self.torch.random.set_rng_state(payload["rng_state"])
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_numer.clear()


__all__ = [
    "CandidateResetManager",
    "MicroBatchPlan",
    "MicroUpdateSession",
    "MicroUpdateTrainer",
    "NO_SUPPORT",
    "OBJECTIVE_FROZEN_PROJECTED_GAP",
    "StepRecord",
    "StudentBackend",
    "TorchStudentBackend",
    "TrainContractError",
    "TrainRunResult",
    "checksum_payload",
    "effective_weight_sum",
    "frozen_projected_gap_numerator",
    "plan_micro_batches",
    "scale_c1_loss",
]
