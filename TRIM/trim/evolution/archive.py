"""Candidate archives. The champion must come from the distillable archive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    content_hash: str
    source: str
    lineage: list[str]
    per_question: dict[str, Any]
    cost: dict[str, Any]
    trigger_stats: dict[str, Any]
    reject_reason: str = ""
    distillable: bool = False
    quality: float = 0.0
    proposal_type: str = "add"
    active_components: int = 0
    coverage: float = 0.0
    protocol_ok: bool = True
    retention_ok: bool = True
    projection_complete: bool = True
    used_unavailable_info: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "source": self.source,
            "lineage": list(self.lineage),
            "per_question": self.per_question,
            "cost": self.cost,
            "trigger_stats": self.trigger_stats,
            "reject_reason": self.reject_reason,
            "distillable": self.distillable,
            "quality": self.quality,
            "proposal_type": self.proposal_type,
            "active_components": self.active_components,
            "coverage": self.coverage,
            "protocol_ok": self.protocol_ok,
            "retention_ok": self.retention_ok,
            "projection_complete": self.projection_complete,
            "used_unavailable_info": self.used_unavailable_info,
        }


@dataclass
class Archive:
    runtime: dict[str, CandidateRecord] = field(default_factory=dict)
    distillable: dict[str, CandidateRecord] = field(default_factory=dict)

    def add(self, record: CandidateRecord) -> None:
        self.runtime[record.content_hash] = record
        if record.distillable and not record.reject_reason:
            self.distillable[record.content_hash] = record

    def reject(self, record: CandidateRecord, reason: str) -> None:
        record.reject_reason = reason
        record.distillable = False
        self.runtime[record.content_hash] = record
        self.distillable.pop(record.content_hash, None)
