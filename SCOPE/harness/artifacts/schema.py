"""Typed PrivilegedArtifact / LocalDecisionArtifact schema (V3).

Name PrivilegedArtifact is retained for BC; semantics are LocalDecisionArtifact.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness.capability.action_space import CapabilityAction
from harness.capability.capability_id import (
    CapabilityId,
    default_module_for,
    parse_capability_id,
)
from harness.artifacts.reason_codes import (
    is_valid_reason_code,
    normalize_reason_code,
)

SCHEMA_VERSION = "scope.artifact.v3"
SCHEMA_VERSION_V1 = "privileged_artifact.v1"


class GuidanceMode(str, Enum):
    ENDORSE = "endorse"
    CORRECT = "correct"
    IGNORE = "ignore"


@dataclass(frozen=True)
class PrivilegedArtifact:
    """Local decision artifact (V3) with V1 compatibility fields."""

    artifact_id: str
    schema_version: str

    episode_id: str
    turn_id: int
    module_id: str

    mode: GuidanceMode
    target_claim_id: str | None
    reason_code: str

    student_action: CapabilityAction
    recommended_action: CapabilityAction | None

    evidence_ids: tuple[str, ...]
    document_ids: tuple[str, ...]

    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- V3 fields ---
    capability_id: str = ""
    target: str | None = None
    diagnosis: str = ""
    recommended_operation: str = ""
    operation_args: dict[str, Any] = field(default_factory=dict)
    runtime_fields_used: tuple[str, ...] = ()
    teacher_source: str = ""
    debug_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "turn_id": self.turn_id,
            "module_id": self.module_id,
            "capability_id": self.capability_id,
            "mode": self.mode.value,
            "target": self.target,
            "target_claim_id": self.target_claim_id,
            "diagnosis": self.diagnosis,
            "recommended_operation": self.recommended_operation,
            "operation_args": dict(self.operation_args),
            "reason_code": self.reason_code,
            "student_action": self.student_action.to_dict(),
            "recommended_action": (
                self.recommended_action.to_dict() if self.recommended_action else None
            ),
            "evidence_ids": list(self.evidence_ids),
            "document_ids": list(self.document_ids),
            "runtime_fields_used": list(self.runtime_fields_used),
            "confidence": self.confidence,
            "teacher_source": self.teacher_source,
            "debug_reason": self.debug_reason,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    def artifact_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def resolved_capability(self) -> CapabilityId:
        if self.capability_id:
            return parse_capability_id(self.capability_id)
        from harness.capability.capability_id import REASON_CODE_TO_CAPABILITY

        return REASON_CODE_TO_CAPABILITY.get(
            self.reason_code, CapabilityId.UNKNOWN
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrivilegedArtifact":
        rec = data.get("recommended_action")
        return cls(
            artifact_id=str(data["artifact_id"]),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            episode_id=str(data["episode_id"]),
            turn_id=int(data["turn_id"]),
            module_id=str(data["module_id"]),
            capability_id=str(data.get("capability_id", "")),
            mode=GuidanceMode(data["mode"]),
            target=data.get("target"),
            target_claim_id=data.get("target_claim_id"),
            diagnosis=str(data.get("diagnosis", "")),
            recommended_operation=str(data.get("recommended_operation", "")),
            operation_args=dict(data.get("operation_args") or {}),
            reason_code=str(data["reason_code"]),
            student_action=CapabilityAction.from_dict(data["student_action"]),
            recommended_action=CapabilityAction.from_dict(rec) if rec else None,
            evidence_ids=tuple(data.get("evidence_ids", ())),
            document_ids=tuple(data.get("document_ids", ())),
            runtime_fields_used=tuple(data.get("runtime_fields_used", ())),
            confidence=float(data.get("confidence", 0.0)),
            teacher_source=str(data.get("teacher_source", "")),
            debug_reason=str(data.get("debug_reason", "")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def build(
        cls,
        *,
        episode_id: str,
        turn_id: int,
        module_id: str,
        mode: GuidanceMode,
        reason_code: str,
        student_action: CapabilityAction,
        recommended_action: CapabilityAction | None = None,
        target_claim_id: str | None = None,
        evidence_ids: tuple[str, ...] | list[str] = (),
        document_ids: tuple[str, ...] | list[str] = (),
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
        capability_id: str | CapabilityId | None = None,
        target: str | None = None,
        diagnosis: str = "",
        recommended_operation: str = "",
        operation_args: dict[str, Any] | None = None,
        runtime_fields_used: tuple[str, ...] | list[str] = (),
        teacher_source: str = "",
        debug_reason: str = "",
    ) -> "PrivilegedArtifact":
        meta = dict(metadata or {})
        cap = parse_capability_id(capability_id) if capability_id else None
        if cap is None or cap == CapabilityId.UNKNOWN:
            from harness.capability.capability_id import REASON_CODE_TO_CAPABILITY

            cap = REASON_CODE_TO_CAPABILITY.get(reason_code, CapabilityId.UNKNOWN)

        if not is_valid_reason_code(reason_code):
            meta["invalid_reason_code"] = reason_code
            reason_code = (
                "MISSING_DIRECT_EVIDENCE"
                if module_id == "verification"
                else "MISSING_DIRECT_SUPPORT"
            )
            if module_id == "budget_control":
                reason_code = "LOW_INFORMATION_GAIN"
            mode = GuidanceMode.IGNORE
            debug_reason = debug_reason or meta.get("invalid_reason_code", "")

        reason_norm = normalize_reason_code(reason_code, cap if cap != CapabilityId.UNKNOWN else None)

        # Infer operation from recommended_action if not provided
        op = recommended_operation
        op_args = dict(operation_args or {})
        if not op and recommended_action is not None:
            op = recommended_action.action_type.value
            if not op_args:
                op_args = dict(recommended_action.arguments)

        if not diagnosis and reason_code:
            diagnosis = reason_norm

        if not module_id and cap != CapabilityId.UNKNOWN:
            module_id = default_module_for(cap)

        return cls(
            artifact_id=str(uuid.uuid4()),
            schema_version=SCHEMA_VERSION,
            episode_id=episode_id,
            turn_id=turn_id,
            module_id=module_id,
            capability_id=cap.value if cap != CapabilityId.UNKNOWN else str(capability_id or ""),
            mode=mode,
            target=target or target_claim_id,
            target_claim_id=target_claim_id,
            diagnosis=diagnosis,
            recommended_operation=op,
            operation_args=op_args,
            reason_code=reason_code,  # keep legacy uppercase for BC audits
            student_action=student_action,
            recommended_action=recommended_action,
            evidence_ids=tuple(evidence_ids),
            document_ids=tuple(document_ids),
            runtime_fields_used=tuple(runtime_fields_used),
            confidence=float(confidence),
            teacher_source=teacher_source,
            debug_reason=debug_reason,
            metadata=meta,
        )


# Semantic alias
LocalDecisionArtifact = PrivilegedArtifact
