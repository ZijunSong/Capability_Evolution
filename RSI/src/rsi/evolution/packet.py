"""EvolutionPacket: everything the proposer may read for one generation.

Includes the real parent source code (or a retrievable file index), paired
baseline/parent results per domain, real failure trace fragments, regression
summaries of still-solved tasks, component trigger/cost stats, historical
candidate summaries, budgets and the editable scope.  The second generation
must see the first generation's source and *new* feedback
(``test_parent_feedback_continuity``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from rsi.contracts.common import JSONValue, digest_of
from rsi.evolution.diagnosis import GenerationDiagnosis
from rsi.harness.loader import LoadedBank


@dataclass(frozen=True)
class TraceFragment:
    task_id: str
    outcome: str
    events_text: str
    decisions: int
    error_kind: str | None = None


@dataclass(frozen=True)
class EvolutionPacket:
    generation: int
    policy_digest: str
    parent_bank_digest: str
    parent_files: Mapping[str, str]  # relative path -> full source (empty for gen 0)
    baseline_scores: Mapping[str, Mapping[str, float]]  # domain -> task -> score
    parent_scores: Mapping[str, Mapping[str, float]]
    paired_summary: Mapping[str, JSONValue]
    failure_fragments: tuple[TraceFragment, ...]
    success_summaries: tuple[TraceFragment, ...]
    diagnosis: GenerationDiagnosis
    candidate_history: tuple[Mapping[str, JSONValue], ...]
    budgets: Mapping[str, JSONValue]
    editable_scope: str = "bank_only"
    editable_paths: tuple[str, ...] = ("components/", "manifest.json")

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "generation": self.generation,
            "policy_digest": self.policy_digest,
            "parent_bank_digest": self.parent_bank_digest,
            "parent_files": dict(self.parent_files),
            "baseline_scores": {d: dict(t) for d, t in self.baseline_scores.items()},
            "parent_scores": {d: dict(t) for d, t in self.parent_scores.items()},
            "paired_summary": dict(self.paired_summary),
            "failure_fragments": [
                {
                    "task_id": f.task_id,
                    "outcome": f.outcome,
                    "events_text": f.events_text,
                    "decisions": f.decisions,
                    "error_kind": f.error_kind,
                }
                for f in self.failure_fragments
            ],
            "success_summaries": [
                {
                    "task_id": f.task_id,
                    "outcome": f.outcome,
                    "events_text": f.events_text,
                    "decisions": f.decisions,
                }
                for f in self.success_summaries
            ],
            "diagnosis": self.diagnosis.to_json(),
            "candidate_history": [dict(h) for h in self.candidate_history],
            "budgets": dict(self.budgets),
            "editable_scope": self.editable_scope,
            "editable_paths": list(self.editable_paths),
        }

    def digest(self) -> str:
        return digest_of(self.to_json())


def bank_source_files(bank: LoadedBank) -> dict[str, str]:
    """Full source of every component file in the parent bank (for the packet)."""
    out: dict[str, str] = {}
    for comp in bank.components:
        for rel in comp.manifest.files:
            path = bank.root / rel
            if path.is_file():
                out[rel.replace("\\", "/")] = path.read_text("utf-8")
    return out


def build_packet(
    *,
    generation: int,
    policy_digest: str,
    parent_bank: LoadedBank,
    baseline_scores: Mapping[str, Mapping[str, float]],
    parent_scores: Mapping[str, Mapping[str, float]],
    diagnosis: GenerationDiagnosis,
    failure_fragments: Sequence[TraceFragment] = (),
    success_summaries: Sequence[TraceFragment] = (),
    candidate_history: Sequence[Mapping[str, JSONValue]] = (),
    budgets: Mapping[str, JSONValue] | None = None,
) -> EvolutionPacket:
    paired: dict[str, JSONValue] = {}
    for domain in sorted(set(baseline_scores) | set(parent_scores)):
        base = baseline_scores.get(domain, {})
        par = parent_scores.get(domain, {})
        deltas = [par[t] - base[t] for t in par if t in base]
        paired[domain] = {
            "n_paired": len(deltas),
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "n_tasks_parent": len(par),
            "n_tasks_baseline": len(base),
        }
    return EvolutionPacket(
        generation=generation,
        policy_digest=policy_digest,
        parent_bank_digest=parent_bank.bank_digest,
        parent_files=bank_source_files(parent_bank),
        baseline_scores={d: dict(t) for d, t in baseline_scores.items()},
        parent_scores={d: dict(t) for d, t in parent_scores.items()},
        paired_summary=paired,
        failure_fragments=tuple(failure_fragments),
        success_summaries=tuple(success_summaries),
        diagnosis=diagnosis,
        candidate_history=tuple(dict(h) for h in candidate_history),
        budgets=dict(budgets or {}),
    )
