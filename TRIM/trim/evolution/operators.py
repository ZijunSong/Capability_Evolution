"""Edit operators. A normal candidate changes one component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


OPERATORS = ("add", "edit", "merge", "prune", "recombine")


@dataclass
class Proposal:
    proposal_id: str
    proposal_type: str
    bank_dir: str
    content_hash: str
    hypothesis: dict[str, Any]
    adaptation: dict[str, Any]
    examples: dict[str, Any] = field(default_factory=dict)
    expected_cost: dict[str, Any] = field(default_factory=dict)
    touched_components: tuple[str, ...] = ()
    edits: int = 1
    lineage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "bank_dir": self.bank_dir,
            "content_hash": self.content_hash,
            "hypothesis": self.hypothesis,
            "adaptation": self.adaptation,
            "examples": self.examples,
            "expected_cost": self.expected_cost,
            "touched_components": list(self.touched_components),
            "edits": self.edits,
            "lineage": list(self.lineage),
        }


def check_edit_budget(proposal: Proposal, *, max_edits: int) -> str | None:
    if proposal.proposal_type not in OPERATORS:
        return f"unknown proposal type {proposal.proposal_type}"
    limit = max_edits
    if proposal.proposal_type in {"merge", "recombine", "prune"}:
        limit = max(max_edits, len(proposal.touched_components) or 1)
    if proposal.edits > limit:
        return f"proposal edits {proposal.edits} exceed {limit}"
    if proposal.proposal_type == "add" and proposal.edits > max_edits:
        return "add proposals are limited to one component"
    return None
