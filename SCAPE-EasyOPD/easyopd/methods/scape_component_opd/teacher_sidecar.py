from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .state_snapshot import SCAPEStateSnapshot, assert_same_state_before_component_fork, stable_hash


@dataclass(frozen=True)
class TeacherSidecarResult:
    teacher_view_hash: str
    teacher_logprobs: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_view_hash": self.teacher_view_hash,
            "teacher_logprobs": list(self.teacher_logprobs),
            "metadata": dict(self.metadata),
        }


class SameWeightsPrivilegedViewTeacher:
    mode = "same_weights_privileged_view"

    def score_student_tokens(self, *, teacher_view_hash: str, student_token_ids: list[int]) -> TeacherSidecarResult:
        return TeacherSidecarResult(
            teacher_view_hash=teacher_view_hash,
            teacher_logprobs=[0.0 for _ in student_token_ids],
            metadata={"stub": True},
        )


class SCAPETeacherSidecar:
    """Teacher-side helper for SCAPE component OPD."""

    def __init__(self, component_name: str, *, mode: str = "same_weights_privileged_view") -> None:
        self.component_name = component_name
        self.mode = mode

    @staticmethod
    def _stable_json(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _snapshot_from_batch(self, batch: Any) -> SCAPEStateSnapshot | None:
        snapshot = None
        if isinstance(batch, dict):
            snapshot = batch.get("scape_snapshot") or batch.get("snapshot")
        else:
            snapshot = getattr(batch, "scape_snapshot", None) or getattr(batch, "snapshot", None)
        if snapshot is None:
            return None
        if isinstance(snapshot, SCAPEStateSnapshot):
            return snapshot
        if isinstance(snapshot, dict):
            return SCAPEStateSnapshot(
                query_id=str(snapshot.get("query_id", "q")),
                turn_id=int(snapshot.get("turn_id", 0)),
                working_memory=dict(snapshot.get("working_memory") or {}),
                documents=list(snapshot.get("documents") or []),
                curated_ids=list(snapshot.get("curated_ids") or []),
                curated_importance=dict(snapshot.get("curated_importance") or {}),
                tool_history=list(snapshot.get("tool_history") or []),
                remaining_budget=snapshot.get("remaining_budget"),
                component_masks=dict(snapshot.get("component_masks") or {}),
                retrieval_state=dict(snapshot.get("retrieval_state") or {}),
                evidence_state=dict(snapshot.get("evidence_state") or {}),
                verified_state=dict(snapshot.get("verified_state") or {}),
            )
        return None

    def build_teacher_view(self, batch: Any, *, config: Any | None = None) -> dict[str, Any]:
        snapshot = self._snapshot_from_batch(batch)
        state_hashes = {}
        if snapshot is not None:
            state_hashes = assert_same_state_before_component_fork(snapshot)
        student_token_ids: list[int] = []
        response_mask: list[Any] = []
        if isinstance(batch, dict):
            student_token_ids = list(batch.get("student_response_token_ids") or batch.get("response_token_ids") or [])
            response_mask = list(batch.get("response_mask") or [])
        else:
            student_token_ids = list(getattr(batch, "student_response_token_ids", None) or getattr(batch, "response_token_ids", None) or [])
            response_mask = list(getattr(batch, "response_mask", None) or [])
        teacher_view = {
            "component": self.component_name,
            "mode": self.mode,
            "state_hashes": state_hashes,
            "student_token_ids": student_token_ids,
            "response_mask": response_mask,
            "teacher_view_hash": stable_hash(
                {
                    "component": self.component_name,
                    "mode": self.mode,
                    "state_hashes": state_hashes,
                    "student_token_ids": student_token_ids,
                    "response_mask": response_mask,
                }
            ),
        }
        if config is not None:
            teacher_view["config"] = config
        return teacher_view

    def teacher_forward(self, batch: Any, teacher_model: Any, *, config: Any | None = None) -> dict[str, Any]:
        teacher_view = self.build_teacher_view(batch, config=config)
        student_token_ids = list(teacher_view.get("student_token_ids") or [])
        if hasattr(teacher_model, "score_student_tokens") and callable(getattr(teacher_model, "score_student_tokens")):
            score = teacher_model.score_student_tokens(
                teacher_view_hash=str(teacher_view["teacher_view_hash"]),
                student_token_ids=student_token_ids,
            )
            if isinstance(score, TeacherSidecarResult):
                payload = score.to_dict()
            elif isinstance(score, dict):
                payload = dict(score)
            else:
                payload = {
                    "teacher_view_hash": str(teacher_view["teacher_view_hash"]),
                    "teacher_logprobs": list(score or []),
                }
        else:
            payload = {
                "teacher_view_hash": str(teacher_view["teacher_view_hash"]),
                "teacher_logprobs": [0.0 for _ in student_token_ids],
            }
        payload.setdefault("metadata", {})
        payload["metadata"].update(
            {
                "component": self.component_name,
                "mode": self.mode,
                "student_token_count": len(student_token_ids),
            }
        )
        payload["teacher_view"] = teacher_view
        return payload
