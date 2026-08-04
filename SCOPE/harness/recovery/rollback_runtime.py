"""Hard rollback executor with invariant enforcement (X5, I1–I8)."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from harness.capability.rollback_operation import RollbackOperation
from harness.recovery.checkpoint_store import CheckpointStore
from harness.recovery.recovery_budget import RecoveryBudget
from harness.telemetry.state_hash import hash_working_memory_fields


@dataclass
class RollbackTelemetryEvent:
    event_id: str
    operation: str
    checkpoint_id: str | None
    before_hash: str
    after_hash: str
    success: bool
    error: str = ""


@dataclass
class RollbackRuntime:
    checkpoint_store: CheckpointStore
    budget: RecoveryBudget = field(default_factory=RecoveryBudget)
    telemetry: list[RollbackTelemetryEvent] = field(default_factory=list)
    invalid_operations: list[str] = field(default_factory=list)
    hidden_fallbacks: int = 0

    def _env_hash(self, env: Any) -> str:
        wm = env.wm
        return hash_working_memory_fields(
            curated_ids=wm.curated_ids,
            pool_ids=wm.pool_ids,
            search_history=wm.search_history,
            observation_ids=[r.get("observation_id") for r in wm.observation_lineage],
            turn_number=wm.turn_number,
        )

    def _restore_env(self, env: Any, snapshot: dict[str, Any]) -> None:
        wm = env.wm
        wm.turn_number = int(snapshot["turn_number"])
        wm.curated_ids = list(snapshot["curated_ids"])
        wm.curated_notes = dict(snapshot["curated_notes"])
        wm.pool_ids = list(snapshot["pool_ids"])
        wm.pool_id_set = set(snapshot.get("pool_id_set", snapshot["pool_ids"]))
        wm.search_history = list(snapshot["search_history"])
        wm.observation_lineage = copy.deepcopy(snapshot["observation_lineage"])
        wm.claim_states = copy.deepcopy(snapshot["claim_states"])
        wm.curated_observation_ids = copy.deepcopy(snapshot["curated_observation_ids"])
        wm.verification_records = copy.deepcopy(snapshot["verification_records"])
        allowed = set(snapshot.get("doc_store_keys", []))
        wm.doc_store = {k: v for k, v in wm.doc_store.items() if k in allowed}

    def execute(
        self,
        env: Any,
        operation: RollbackOperation,
        *,
        checkpoint_id: str | None = None,
    ) -> bool:
        event_id = f"rb_{uuid.uuid4().hex[:10]}"
        before = self._env_hash(env)

        if operation in (RollbackOperation.CONTINUE, RollbackOperation.REPLAN):
            self.telemetry.append(
                RollbackTelemetryEvent(
                    event_id=event_id,
                    operation=operation.value,
                    checkpoint_id=None,
                    before_hash=before,
                    after_hash=before,
                    success=True,
                )
            )
            return True

        if operation != RollbackOperation.ROLLBACK_TO:
            self.invalid_operations.append(f"illegal_op:{operation}")
            raise ValueError(f"illegal rollback operation: {operation}")

        if not checkpoint_id or not self.checkpoint_store.exists(checkpoint_id):
            self.invalid_operations.append(f"missing_checkpoint:{checkpoint_id}")
            raise ValueError(f"checkpoint_id must exist: {checkpoint_id}")

        if not self.budget.can_rollback():
            self.invalid_operations.append(f"budget:{event_id}")
            raise RuntimeError("rollback budget exhausted")

        cp = self.checkpoint_store.get(checkpoint_id)
        if cp is None:
            raise ValueError(f"checkpoint missing: {checkpoint_id}")

        self.budget.consume(event_id)
        self._restore_env(env, cp.snapshot)
        after = self._env_hash(env)

        if after != cp.state_hash:
            self.invalid_operations.append(f"hash_mismatch:{checkpoint_id}")
            raise RuntimeError(
                f"state hash mismatch after rollback: {after} != {cp.state_hash}"
            )

        self.telemetry.append(
            RollbackTelemetryEvent(
                event_id=event_id,
                operation=RollbackOperation.ROLLBACK_TO.value,
                checkpoint_id=checkpoint_id,
                before_hash=before,
                after_hash=after,
                success=True,
            )
        )
        return True
