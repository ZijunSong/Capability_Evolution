"""Self-evolution proposer. It reads failure counts, not gold answers, and does not score itself."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from trim.evolution.materialize import parse_proposal_text, write_candidate_bank
from trim.evolution.operators import Proposal

PROMPT = """You are editing one retrieval harness component. The student model and the H0 tools stay fixed.
Reply with one JSON object and no other text. Keys:
failure_type, trajectory_id, mechanism, expected_fix, rollback, guidance.
failure_type must be one of bad_query, unread_evidence, redundant_search, bad_selection, early_stop, context_overflow, format_or_tool_error.
guidance is a general instruction that uses only the visible question and search history.
Do not include a query id, a document id, a gold answer, or a question-specific search string.
"""


class SelfHarnessProposer:
    def __init__(
        self,
        *,
        generate_text: Callable[[str], str],
        work_dir: Path,
        failure_counts: Mapping[str, int],
        trajectory_id: str,
    ) -> None:
        self.generate_text = generate_text
        self.work_dir = work_dir
        self.failure_counts = dict(failure_counts)
        self.trajectory_id = trajectory_id

    def propose(self, packet: dict[str, Any]) -> list[Proposal]:
        generation = int(packet.get("generation") or 0)
        prompt = (
            PROMPT
            + "\nFailure counts from the frozen H0 rollout:\n"
            + "\n".join(f"{name}: {count}" for name, count in sorted(self.failure_counts.items()))
            + f"\nRepresentative trajectory_id: {self.trajectory_id}\n"
        )
        try:
            payload = parse_proposal_text(self.generate_text(prompt))
        except Exception:
            return []
        payload["trajectory_id"] = self.trajectory_id
        root = self.work_dir / f"g{generation:03d}" / "bank"
        kind = "prune" if packet.get("want_prune") else ("edit" if generation else "add")
        if kind == "prune":
            payload = dict(payload)
            payload["guidance"] = "Do not add a search beyond the visible question."
        try:
            return [write_candidate_bank(root, payload, proposal_type="edit" if kind == "prune" else kind)]
        except Exception:
            return []
