from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SCAPEStateSnapshot:
    query_id: str
    turn_id: int
    working_memory: dict[str, Any] = field(default_factory=dict)
    documents: list[dict[str, Any]] = field(default_factory=list)
    curated_ids: list[str] = field(default_factory=list)
    curated_importance: dict[str, str] = field(default_factory=dict)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    remaining_budget: int | None = None
    component_masks: dict[str, bool] = field(default_factory=dict)
    retrieval_state: dict[str, Any] = field(default_factory=dict)
    evidence_state: dict[str, Any] = field(default_factory=dict)
    verified_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "turn_id": self.turn_id,
            "working_memory": copy.deepcopy(self.working_memory),
            "documents": copy.deepcopy(self.documents),
            "curated_ids": list(self.curated_ids),
            "curated_importance": dict(self.curated_importance),
            "tool_history": copy.deepcopy(self.tool_history),
            "remaining_budget": self.remaining_budget,
            "component_masks": dict(self.component_masks),
            "retrieval_state": copy.deepcopy(self.retrieval_state),
            "evidence_state": copy.deepcopy(self.evidence_state),
            "verified_state": copy.deepcopy(self.verified_state),
        }

    def state_hash(self) -> str:
        return stable_hash(self.to_dict())

    def restore(self) -> "SCAPEStateSnapshot":
        return SCAPEStateSnapshot(**self.to_dict())


def assert_same_state_before_component_fork(snapshot: SCAPEStateSnapshot) -> dict[str, str]:
    student_branch = snapshot.restore()
    teacher_branch = snapshot.restore()
    student_hash = student_branch.state_hash()
    teacher_hash = teacher_branch.state_hash()
    if student_hash != teacher_hash:
        raise AssertionError("student_branch.state_hash != teacher_branch.state_hash before component effect")
    return {"state_hash_pre": snapshot.state_hash(), "state_hash_student": student_hash, "state_hash_teacher": teacher_hash}
