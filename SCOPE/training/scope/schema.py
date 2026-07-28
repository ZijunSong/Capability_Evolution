"""SCOPE v3 DecisionSupervisionSample schema."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness.artifacts.schema import PrivilegedArtifact
from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState

SCHEMA_VERSION = "scope.supervision.v3"


class BranchType(str, Enum):
    MAIN = "MAIN"
    RECOVERY = "RECOVERY"


class Route(str, Enum):
    ENDORSE = "ENDORSE"
    CORRECT = "CORRECT"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class GateFlags:
    visible: bool = True
    schema_valid: bool = True
    module_valid: bool = True
    executable: bool = True
    provenance_ok: bool = True
    purity_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": self.visible,
            "schema_valid": self.schema_valid,
            "module_valid": self.module_valid,
            "executable": self.executable,
            "provenance_ok": self.provenance_ok,
            "purity_ok": self.purity_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GateFlags":
        data = data or {}
        return cls(
            visible=bool(data.get("visible", True)),
            schema_valid=bool(data.get("schema_valid", True)),
            module_valid=bool(data.get("module_valid", True)),
            executable=bool(data.get("executable", True)),
            provenance_ok=bool(data.get("provenance_ok", True)),
            purity_ok=bool(data.get("purity_ok", True)),
        )

    @property
    def all_passed(self) -> bool:
        return all(
            [
                self.visible,
                self.schema_valid,
                self.module_valid,
                self.executable,
                self.provenance_ok,
                self.purity_ok,
            ]
        )


@dataclass(frozen=True)
class VerificationFlags:
    student_valid: bool | None = None
    target_valid: bool | None = None
    score_student: float | None = None
    score_target: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_valid": self.student_valid,
            "target_valid": self.target_valid,
            "score_student": self.score_student,
            "score_target": self.score_target,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VerificationFlags":
        data = data or {}
        return cls(
            student_valid=data.get("student_valid"),
            target_valid=data.get("target_valid"),
            score_student=data.get("score_student"),
            score_target=data.get("score_target"),
        )


@dataclass(frozen=True)
class WeightTerms:
    procedural_purity: float = 1.0
    reliability: float = 1.0
    internalization: float = 0.0
    local_gain: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedural_purity": self.procedural_purity,
            "reliability": self.reliability,
            "internalization": self.internalization,
            "local_gain": self.local_gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WeightTerms":
        data = data or {}
        return cls(
            procedural_purity=float(data.get("procedural_purity", 1.0)),
            reliability=float(data.get("reliability", 1.0)),
            internalization=float(data.get("internalization", 0.0)),
            local_gain=float(data.get("local_gain", 1.0)),
        )


@dataclass(frozen=True)
class DecisionSupervisionSampleV3:
    sample_id: str
    schema_version: str

    episode_id: str
    event_id: str
    turn: int
    task_id: str

    branch_type: BranchType
    capability_id: str
    module_id: str

    decision_state: dict[str, Any]
    student_action: dict[str, Any]
    target_action: dict[str, Any] | None

    route: Route
    artifact: dict[str, Any]

    gates: GateFlags
    verification: VerificationFlags
    weight_terms: WeightTerms
    sample_weight: float

    # Training masks
    train_mask: int = 1
    loss_mask_action_only: bool = True

    # Text forms for CE loss (optional pre-rendered)
    student_state_text: str = ""
    target_action_text: str = ""

    audit_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "event_id": self.event_id,
            "turn": self.turn,
            "task_id": self.task_id,
            "branch_type": self.branch_type.value,
            "capability_id": self.capability_id,
            "module_id": self.module_id,
            "decision_state": dict(self.decision_state),
            "student_action": dict(self.student_action),
            "target_action": dict(self.target_action) if self.target_action else None,
            "route": self.route.value,
            "artifact": dict(self.artifact),
            "gates": self.gates.to_dict(),
            "verification": self.verification.to_dict(),
            "weight_terms": self.weight_terms.to_dict(),
            "sample_weight": self.sample_weight,
            "train_mask": self.train_mask,
            "loss_mask_action_only": self.loss_mask_action_only,
            "student_state_text": self.student_state_text,
            "target_action_text": self.target_action_text,
            "audit_error": self.audit_error,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    def sample_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionSupervisionSampleV3":
        return cls(
            sample_id=str(data.get("sample_id") or uuid.uuid4()),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            episode_id=str(data["episode_id"]),
            event_id=str(data.get("event_id", "")),
            turn=int(data.get("turn", data.get("turn_id", 0))),
            task_id=str(data.get("task_id", "")),
            branch_type=BranchType(data.get("branch_type", "MAIN")),
            capability_id=str(data.get("capability_id", "")),
            module_id=str(data.get("module_id", "")),
            decision_state=dict(data.get("decision_state") or {}),
            student_action=dict(data.get("student_action") or {}),
            target_action=dict(data["target_action"]) if data.get("target_action") else None,
            route=Route(str(data.get("route", "IGNORE")).upper()),
            artifact=dict(data.get("artifact") or {}),
            gates=GateFlags.from_dict(data.get("gates")),
            verification=VerificationFlags.from_dict(data.get("verification")),
            weight_terms=WeightTerms.from_dict(data.get("weight_terms")),
            sample_weight=float(data.get("sample_weight", 1.0)),
            train_mask=int(data.get("train_mask", 1)),
            loss_mask_action_only=bool(data.get("loss_mask_action_only", True)),
            student_state_text=str(data.get("student_state_text", "")),
            target_action_text=str(data.get("target_action_text", "")),
            audit_error=data.get("audit_error"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def build(
        cls,
        *,
        state: DecisionState,
        artifact: PrivilegedArtifact,
        student_action: CapabilityAction,
        target_action: CapabilityAction | None,
        route: Route,
        gates: GateFlags,
        verification: VerificationFlags | None = None,
        weight_terms: WeightTerms | None = None,
        sample_weight: float = 1.0,
        branch_type: BranchType = BranchType.MAIN,
        event_id: str = "",
        train_mask: int = 1,
        student_state_text: str = "",
        target_action_text: str = "",
        audit_error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DecisionSupervisionSampleV3":
        cap = artifact.capability_id or artifact.resolved_capability().value
        return cls(
            sample_id=str(uuid.uuid4()),
            schema_version=SCHEMA_VERSION,
            episode_id=state.episode_id,
            event_id=event_id or state.event_id or f"{state.episode_id}:{state.turn_id}",
            turn=state.turn_id,
            task_id=state.task_id,
            branch_type=branch_type,
            capability_id=cap,
            module_id=artifact.module_id,
            decision_state=state.to_dict(),
            student_action=student_action.to_dict(),
            target_action=target_action.to_dict() if target_action else None,
            route=route,
            artifact=artifact.to_dict(),
            gates=gates,
            verification=verification or VerificationFlags(),
            weight_terms=weight_terms or WeightTerms(),
            sample_weight=float(sample_weight),
            train_mask=int(train_mask),
            student_state_text=student_state_text or state.rendered_context,
            target_action_text=target_action_text,
            audit_error=audit_error,
            metadata=dict(metadata or {}),
        )
