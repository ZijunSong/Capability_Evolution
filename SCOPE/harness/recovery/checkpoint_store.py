"""Checkpoint snapshots for rollback recovery (X4)."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from harness.telemetry.state_hash import hash_working_memory_fields


@dataclass
class Checkpoint:
    checkpoint_id: str
    state_hash: str
    branch_id: str
    turn_id: int
    snapshot: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointStore:
    def __init__(self, branch_id: str = "main") -> None:
        self.branch_id = branch_id
        self._checkpoints: dict[str, Checkpoint] = {}

    def list_ids(self) -> list[str]:
        return sorted(self._checkpoints.keys())

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        return self._checkpoints.get(checkpoint_id)

    def exists(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self._checkpoints

    def save_from_env(self, env: Any, *, turn_id: int, label: str = "") -> Checkpoint:
        wm = env.wm
        snapshot = {
            "turn_number": int(wm.turn_number),
            "curated_ids": list(wm.curated_ids),
            "curated_notes": dict(wm.curated_notes),
            "pool_ids": list(wm.pool_ids),
            "pool_id_set": list(wm.pool_id_set),
            "search_history": list(wm.search_history),
            "observation_lineage": copy.deepcopy(wm.observation_lineage),
            "claim_states": copy.deepcopy(wm.claim_states),
            "curated_observation_ids": copy.deepcopy(wm.curated_observation_ids),
            "verification_records": copy.deepcopy(wm.verification_records),
            "doc_store_keys": sorted(wm.doc_store.keys()),
        }
        state_hash = hash_working_memory_fields(
            curated_ids=snapshot["curated_ids"],
            pool_ids=snapshot["pool_ids"],
            search_history=snapshot["search_history"],
            observation_ids=[
                r.get("observation_id") for r in snapshot["observation_lineage"]
            ],
            turn_number=snapshot["turn_number"],
        )
        cid = f"ckpt_{turn_id}_{uuid.uuid4().hex[:8]}"
        cp = Checkpoint(
            checkpoint_id=cid,
            state_hash=state_hash,
            branch_id=self.branch_id,
            turn_id=turn_id,
            snapshot=snapshot,
            metadata={"label": label, "env_turn": int(getattr(env, "_current_turn", turn_id))},
        )
        self._checkpoints[cid] = cp
        return cp

    def lightweight_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "turn_id": cp.turn_id,
                "state_hash": cp.state_hash,
                "n_curated": len(cp.snapshot.get("curated_ids", [])),
                "n_pool": len(cp.snapshot.get("pool_ids", [])),
            }
            for cp in self._checkpoints.values()
        ]
