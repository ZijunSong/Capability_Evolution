"""OPDTransitionV2 data structure."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact

SCHEMA_VERSION = "opd_transition_v2.v1"


@dataclass(frozen=True)
class OPDTransitionV2:
    transition_id: str

    episode_id: str
    task_id: str
    turn_id: int
    module_id: str
    mode: GuidanceMode
    reason_code: str

    student_state_text: str
    student_action_text: str

    teacher_state_text: str | None
    recommended_action_text: str | None

    student_action_token_ids: tuple[int, ...]
    recommended_action_token_ids: tuple[int, ...] | None

    artifact: PrivilegedArtifact

    validity_mask: int
    teacher_confidence: float

    final_reward: float
    module_weight: float

    policy_version: str
    tokenizer_version: str
    schema_version: str = SCHEMA_VERSION

    wm_snapshot_hash: str = ""
    state_hash: str = ""
    artifact_hash: str = ""
    config_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "module_id": self.module_id,
            "mode": self.mode.value,
            "reason_code": self.reason_code,
            "student_state_text": self.student_state_text,
            "student_action_text": self.student_action_text,
            "teacher_state_text": self.teacher_state_text,
            "recommended_action_text": self.recommended_action_text,
            "student_action_token_ids": list(self.student_action_token_ids),
            "recommended_action_token_ids": (
                list(self.recommended_action_token_ids)
                if self.recommended_action_token_ids is not None
                else None
            ),
            "artifact": self.artifact.to_dict(),
            "validity_mask": self.validity_mask,
            "teacher_confidence": self.teacher_confidence,
            "final_reward": self.final_reward,
            "module_weight": self.module_weight,
            "policy_version": self.policy_version,
            "tokenizer_version": self.tokenizer_version,
            "schema_version": self.schema_version,
            "wm_snapshot_hash": self.wm_snapshot_hash,
            "state_hash": self.state_hash,
            "artifact_hash": self.artifact_hash,
            "config_hash": self.config_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OPDTransitionV2":
        rec = data.get("recommended_action_token_ids")
        return cls(
            transition_id=str(data["transition_id"]),
            episode_id=str(data["episode_id"]),
            task_id=str(data["task_id"]),
            turn_id=int(data["turn_id"]),
            module_id=str(data["module_id"]),
            mode=GuidanceMode(data["mode"]),
            reason_code=str(data["reason_code"]),
            student_state_text=str(data.get("student_state_text", "")),
            student_action_text=str(data.get("student_action_text", "")),
            teacher_state_text=data.get("teacher_state_text"),
            recommended_action_text=data.get("recommended_action_text"),
            student_action_token_ids=tuple(data.get("student_action_token_ids", ())),
            recommended_action_token_ids=tuple(rec) if rec is not None else None,
            artifact=PrivilegedArtifact.from_dict(data["artifact"]),
            validity_mask=int(data.get("validity_mask", 0)),
            teacher_confidence=float(data.get("teacher_confidence", 0.0)),
            final_reward=float(data.get("final_reward", 0.0)),
            module_weight=float(data.get("module_weight", 1.0)),
            policy_version=str(data.get("policy_version", "")),
            tokenizer_version=str(data.get("tokenizer_version", "")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            wm_snapshot_hash=str(data.get("wm_snapshot_hash", "")),
            state_hash=str(data.get("state_hash", "")),
            artifact_hash=str(data.get("artifact_hash", "")),
            config_hash=str(data.get("config_hash", "")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def build(
        cls,
        *,
        episode_id: str,
        task_id: str,
        turn_id: int,
        module_id: str,
        mode: GuidanceMode,
        reason_code: str,
        student_state_text: str,
        student_action_text: str,
        artifact: PrivilegedArtifact,
        teacher_state_text: str | None = None,
        recommended_action_text: str | None = None,
        student_action_token_ids: tuple[int, ...] | list[int] = (),
        recommended_action_token_ids: tuple[int, ...] | list[int] | None = None,
        validity_mask: int = 1,
        teacher_confidence: float = 0.0,
        final_reward: float = 0.0,
        module_weight: float = 1.0,
        policy_version: str = "",
        tokenizer_version: str = "",
        wm_snapshot_hash: str = "",
        state_hash: str = "",
        config_hash: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "OPDTransitionV2":
        return cls(
            transition_id=str(uuid.uuid4()),
            episode_id=episode_id,
            task_id=task_id,
            turn_id=turn_id,
            module_id=module_id,
            mode=mode,
            reason_code=reason_code,
            student_state_text=student_state_text,
            student_action_text=student_action_text,
            teacher_state_text=teacher_state_text,
            recommended_action_text=recommended_action_text,
            student_action_token_ids=tuple(student_action_token_ids),
            recommended_action_token_ids=(
                tuple(recommended_action_token_ids)
                if recommended_action_token_ids is not None
                else None
            ),
            artifact=artifact,
            validity_mask=validity_mask,
            teacher_confidence=teacher_confidence,
            final_reward=final_reward,
            module_weight=module_weight,
            policy_version=policy_version,
            tokenizer_version=tokenizer_version,
            wm_snapshot_hash=wm_snapshot_hash,
            state_hash=state_hash,
            artifact_hash=artifact.artifact_hash(),
            config_hash=config_hash,
            metadata=dict(metadata or {}),
        )


def config_hash_from_dict(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
