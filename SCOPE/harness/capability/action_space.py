"""Structured capability action space for dual-mode OPD."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CapabilityActionType(str, Enum):
    SEARCH = "search"
    REWRITE_QUERY = "rewrite_query"
    OPEN_DOCUMENT = "open_document"
    CURATE_DOCUMENT = "curate_document"
    UPDATE_EVIDENCE = "update_evidence"
    VERIFY_CLAIM = "verify_claim"
    CONTINUE_SEARCH = "continue_search"
    STOP_AND_ANSWER = "stop_and_answer"
    ANSWER = "answer"
    ABSTAIN = "abstain"
    GREP = "grep"
    REVIEW_DOCS = "review_docs"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CapabilityAction:
    action_type: CapabilityActionType
    arguments: dict[str, Any] = field(default_factory=dict)
    target_claim_id: str | None = None
    source_observation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "arguments": dict(self.arguments),
            "target_claim_id": self.target_claim_id,
            "source_observation_ids": list(self.source_observation_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityAction":
        raw_type = data.get("action_type", "unknown")
        try:
            action_type = CapabilityActionType(raw_type)
        except ValueError:
            action_type = CapabilityActionType.UNKNOWN
        return cls(
            action_type=action_type,
            arguments=dict(data.get("arguments", {})),
            target_claim_id=data.get("target_claim_id"),
            source_observation_ids=tuple(data.get("source_observation_ids", ())),
        )

    def canonical_key(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
