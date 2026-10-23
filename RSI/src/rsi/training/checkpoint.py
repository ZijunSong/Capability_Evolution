"""Checkpointing (task book §9.5).

Saves model/adapter, optimizer, scheduler, RNG, sampler position, policy
version, global step, batch identity and bank/H0/environment/data digests.
Mid-run resume restores the optimizer; cross-outer-loop default keeps
accumulated model parameters and resets optimizer/scheduler (explicit config).

Sharded, manifest'd JSONL/Parquet for rows; big snapshots are content-addressed
separately — never a single tens-of-GB JSON fully loaded per rank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.errors import ProtocolViolation

CHECKPOINT_SCHEMA = "rsi_checkpoint_v1"


@dataclass(frozen=True)
class CheckpointIdentity:
    policy_digest: str
    bank_digest: str
    runtime_contract_digest: str
    environment_digest: str
    data_digest: str
    global_step: int
    batch_identity: str
    method: str

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "policy_digest": self.policy_digest,
            "bank_digest": self.bank_digest,
            "runtime_contract_digest": self.runtime_contract_digest,
            "environment_digest": self.environment_digest,
            "data_digest": self.data_digest,
            "global_step": self.global_step,
            "batch_identity": self.batch_identity,
            "method": self.method,
        }


def save_checkpoint(
    directory: str | Path,
    *,
    backend: Any,
    identity: CheckpointIdentity,
    rng_state: Any | None = None,
    sampler_position: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    import torch

    model_state = {k: v.detach().cpu() for k, v in backend.model.state_dict().items()}
    torch.save(model_state, directory / "model.pt")
    if getattr(backend, "_optimizer", None) is not None:
        torch.save(backend._optimizer.state_dict(), directory / "optimizer.pt")
    if rng_state is not None:
        torch.save(rng_state, directory / "rng.pt")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity.to_json(),
        "sampler_position": sampler_position,
        "extra": dict(extra or {}),
        "model_digest": digest_of({k: [float(x) for x in v.flatten().tolist()[:8]] for k, v in sorted(model_state.items())}),
    }
    (directory / "checkpoint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8"
    )
    return directory


def load_checkpoint(
    directory: str | Path,
    *,
    backend: Any,
    restore_optimizer: bool = True,
) -> dict[str, Any]:
    directory = Path(directory)
    ckpt_path = directory / "checkpoint.json"
    if not ckpt_path.is_file():
        raise ProtocolViolation(f"checkpoint manifest missing: {ckpt_path}")
    payload = json.loads(ckpt_path.read_text("utf-8"))
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ProtocolViolation("unknown checkpoint schema")
    import torch

    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    backend.model.load_state_dict(state)
    if restore_optimizer and (directory / "optimizer.pt").is_file() and getattr(backend, "_optimizer", None) is not None:
        backend._optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True))
    return payload
