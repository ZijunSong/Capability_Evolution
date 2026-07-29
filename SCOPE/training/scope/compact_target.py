"""Compact Dup capability target: operation-level supervision for Round 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from harness.capability.action_space import CapabilityAction, CapabilityActionType


from harness.capability.dup_operation import DupOperation


@dataclass(frozen=True)
class CompactDupTarget:
    operation: DupOperation
    candidate_id: str | None = None
    canonical_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"operation": self.operation.value}
        if self.candidate_id:
            out["candidate_id"] = self.candidate_id
        if self.canonical_id:
            out["canonical_id"] = self.canonical_id
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompactDupTarget":
        op = DupOperation(str(data["operation"]).upper())
        return cls(
            operation=op,
            candidate_id=data.get("candidate_id"),
            canonical_id=data.get("canonical_id"),
        )


def infer_operation_from_action(action: dict[str, Any] | None) -> DupOperation | None:
    """Map runtime action JSON to compact Dup operation."""
    if not action:
        return None
    if "operation" in action:
        try:
            return DupOperation(str(action["operation"]).upper())
        except ValueError:
            return None
    cap = CapabilityAction.from_dict(action)
    if cap.action_type != CapabilityActionType.CURATE_DOCUMENT:
        return None
    adds = cap.arguments.get("add_ids") or []
    rems = cap.arguments.get("remove_ids") or []
    if not adds and not rems:
        return DupOperation.SKIP_DUPLICATE
    if not adds and rems:
        return DupOperation.SKIP_DUPLICATE
    return DupOperation.KEEP_EVIDENCE


def infer_operation_from_artifact(artifact: dict[str, Any]) -> DupOperation | None:
    op = str(artifact.get("recommended_operation") or "").lower()
    if "skip" in op:
        return DupOperation.SKIP_DUPLICATE
    if "keep" in op or "curate" in op:
        return DupOperation.KEEP_EVIDENCE
    reason = str(artifact.get("reason_code") or "").upper()
    if reason == "DUPLICATE_EVIDENCE":
        return DupOperation.SKIP_DUPLICATE
    return None


def compact_target_from_sample(sample: dict[str, Any]) -> CompactDupTarget | None:
    meta = sample.get("metadata") or {}
    if "compact_target" in meta:
        return CompactDupTarget.from_dict(meta["compact_target"])
    target = sample.get("target_action")
    op = infer_operation_from_action(target)
    if op is None:
        art = sample.get("artifact") or {}
        op = infer_operation_from_artifact(art)
    if op is None:
        return None
    candidate_id = None
    canonical_id = None
    art = sample.get("artifact") or {}
    tgt_action = sample.get("target_action")
    target_id = art.get("target")
    if not target_id and isinstance(tgt_action, dict):
        adds = (tgt_action.get("arguments") or {}).get("add_ids") or []
        target_id = adds[0] if adds else None
    if isinstance(target_id, str):
        candidate_id = target_id
    return CompactDupTarget(operation=op, candidate_id=candidate_id, canonical_id=canonical_id)


def render_compact_target(target: CompactDupTarget) -> str:
    """Short JSON string for CE loss — operation tokens dominate."""
    return target.to_json()


def apply_compact_target_to_sample(
    sample: dict[str, Any],
    *,
    mask_arguments: bool = True,
) -> dict[str, Any]:
    """Return copy with compact target_action and metadata."""
    compact = compact_target_from_sample(sample)
    if compact is None:
        return sample
    out = dict(sample)
    out["target_action"] = compact.to_dict()
    meta = dict(out.get("metadata") or {})
    meta["compact_target"] = compact.to_dict()
    meta["target_format"] = "compact_operation"
    meta["mask_argument_tokens"] = mask_arguments
    out["metadata"] = meta
    out["target_action_text"] = render_compact_target(compact)
    return out
