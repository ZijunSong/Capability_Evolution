"""One generation of harness code evolution (task book §7.2).

Parent selection -> packet -> real proposer request -> materialize -> validate
-> dev-panel screening -> common selection-panel retest (candidate vs current
baseline, same tasks/seeds) -> runtime archive -> (champion-only) real OPD
probe -> frozen champion for confirmation.

Repair is limited (``max_candidate_repairs``); parse/validation failures are
recorded and fed back, never silently fabricated away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rsi.contracts.artifact import CandidatePatch
from rsi.contracts.common import JSONValue
from rsi.contracts.errors import CandidateInvalid
from rsi.evolution.archive import Archive, DistillationEntry, RuntimeEntry  # noqa: F401
from rsi.evolution.diagnosis import GenerationDiagnosis, diagnose_traces
from rsi.evolution.materializer import MaterializedCandidate, materialize_candidate
from rsi.evolution.packet import EvolutionPacket, TraceFragment, build_packet
from rsi.evolution.proposer import Proposer
from rsi.evolution.selection import CandidateScore, CostVector, SelectionResult, select_champion
from rsi.evolution.validator import validate_materialized
from rsi.harness.loader import LoadedBank, load_bank, load_empty_bank


@dataclass
class EvolutionBudgets:
    max_generations: int = 8
    candidates_per_generation: int = 2
    max_candidate_evaluations: int = 16
    plateau_patience: int = 3
    max_candidate_repairs: int = 2


@dataclass
class GenerationRecord:
    generation: int
    parent_bank_digest: str
    candidate_ids: tuple[str, ...]
    selection: SelectionResult
    diagnosis: GenerationDiagnosis
    champion_id: str | None
    champion_bank_digest: str | None
    stopped_reason: str | None = None

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "generation": self.generation,
            "parent_bank_digest": self.parent_bank_digest,
            "candidate_ids": list(self.candidate_ids),
            "selection": self.selection.to_json(),
            "diagnosis": self.diagnosis.to_json(),
            "champion_id": self.champion_id,
            "champion_bank_digest": self.champion_bank_digest,
            "stopped_reason": self.stopped_reason,
        }


@dataclass
class EvolveResult:
    champion: MaterializedCandidate | None
    generations: tuple[GenerationRecord, ...]
    stopped_reason: str
    archive_root: str
    evaluated_candidates: int

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "ok": self.champion is not None,
            "champion": (
                {
                    "candidate_id": self.champion.candidate_id,
                    "bank_dir": str(self.champion.bank_dir),
                    "bank_digest": self.champion.bank_digest,
                    "hypothesis": self.champion.hypothesis,
                }
                if self.champion
                else None
            ),
            "generations": [g.to_json() for g in self.generations],
            "stopped_reason": self.stopped_reason,
            "archive_root": self.archive_root,
            "evaluated_candidates": self.evaluated_candidates,
        }


PanelRunner = Callable[[LoadedBank], Mapping[str, Any]]
# PanelRunner must return {"scores": {domain: {task: value}}, "per_task": {...},
# "cost": {...}} for a bank evaluated on the dev/selection panel.


class EvolutionSearch:
    def __init__(
        self,
        *,
        proposer: Proposer,
        archive: Archive,
        panel_runner: PanelRunner,
        packet_builder: Callable[..., EvolutionPacket],
        budgets: EvolutionBudgets | None = None,
        parent_bank_root: Path | None = None,
        candidates_root: Path | None = None,
        known_task_ids: Sequence[str] = (),
        baseline_scores: Mapping[str, Mapping[str, float]] | None = None,
        seed_mode: str = "empty_each_round",
    ):
        self.proposer = proposer
        self.archive = archive
        self.panel_runner = panel_runner
        self.packet_builder = packet_builder
        self.budgets = budgets or EvolutionBudgets()
        self.parent_bank_root = parent_bank_root
        self.candidates_root = candidates_root or (archive.root.parent / "candidates")
        self.candidates_root.mkdir(parents=True, exist_ok=True)
        self.known_task_ids = tuple(known_task_ids)
        self.seed_mode = seed_mode
        self._baseline_scores = {d: dict(t) for d, t in (baseline_scores or {}).items()}
        self._history: list[dict[str, JSONValue]] = []
        self._evaluated = 0

    def run(self) -> EvolveResult:
        generations: list[GenerationRecord] = []
        # Round seed: empty_each_round starts from the empty bank every round;
        # warm_archive allows old source as parents but still re-measures.
        if self.seed_mode == "empty_each_round" or self.parent_bank_root is None:
            parent: LoadedBank = load_empty_bank()
        else:
            parent = load_bank(self.parent_bank_root)

        champion: MaterializedCandidate | None = None
        best_macro: float | None = None
        stale = 0
        stopped_reason = "MAX_GENERATIONS"

        for gen in range(self.budgets.max_generations):
            parent_result = self.panel_runner(parent)
            parent_scores = dict(parent_result["scores"])
            diagnosis = diagnose_traces(
                generation=gen,
                per_task=dict(parent_result.get("per_task") or {}),
                baseline_scores=None,
                parent_scores=None,
            )
            packet = self.packet_builder(
                generation=gen,
                policy_digest=str(parent_result.get("policy_digest", "")),
                parent_bank=parent,
                baseline_scores=self._baseline_scores or parent_scores,
                parent_scores=parent_scores,
                diagnosis=diagnosis,
                candidate_history=tuple(self._history),
                budgets={
                    "max_generations": self.budgets.max_generations,
                    "candidates_per_generation": self.budgets.candidates_per_generation,
                    "max_candidate_evaluations": self.budgets.max_candidate_evaluations,
                },
            )

            gen_candidates: list[tuple[MaterializedCandidate, CandidateScore]] = []
            candidate_ids: list[str] = []
            for _slot in range(self.budgets.candidates_per_generation):
                if self._evaluated >= self.budgets.max_candidate_evaluations:
                    break
                mat = self._propose_materialize(packet, parent)
                if mat is None:
                    continue
                candidate_ids.append(mat.candidate_id)
                score = self._score_candidate(mat)
                gen_candidates.append((mat, score))
                self.archive.append_runtime(
                    RuntimeEntry(
                        candidate_id=mat.candidate_id,
                        bank_digest=mat.bank_digest,
                        parent_bank_digest=mat.parent_bank_digest,
                        generation=gen,
                        operation=mat.operation,
                        hypothesis=mat.hypothesis,
                        scores={d: {"__mean__": v} for d, v in score.per_domain.items()},
                        macro_score=score.macro_score,
                        cost=score.cost.to_json(),  # type: ignore[arg-type]
                        lineage=(mat.parent_bank_digest,),
                        status="legal",
                    )
                )
                self._history.append(
                    {
                        "candidate_id": mat.candidate_id,
                        "generation": gen,
                        "hypothesis": mat.hypothesis,
                        "macro_score": score.macro_score,
                    }
                )

            if not gen_candidates:
                record = GenerationRecord(
                    generation=gen,
                    parent_bank_digest=parent.bank_digest,
                    candidate_ids=(),
                    selection=SelectionResult(champion_id=None, reason="NO_VALID_CANDIDATES", ordered=(), tolerance=0.01),
                    diagnosis=diagnosis,
                    champion_id=champion.candidate_id if champion else None,
                    champion_bank_digest=champion.bank_digest if champion else None,
                    stopped_reason="CANDIDATES_REPEATEDLY_INVALID",
                )
                generations.append(record)
                stale += 1
                if stale >= self.budgets.plateau_patience:
                    stopped_reason = "CANDIDATES_REPEATEDLY_INVALID"
                    break
                continue

            baseline_score = self._as_candidate_score(parent, parent_result)
            scores = [s for _, s in gen_candidates]
            selection = select_champion(scores, baseline=baseline_score)
            if selection.champion_id is None:
                winner = None
            else:
                winner = next((m for m, s in gen_candidates if m.candidate_id == selection.champion_id), None)

            improved = False
            if winner is not None:
                win_score = next(s for m, s in gen_candidates if m.candidate_id == winner.candidate_id)
                if best_macro is None or win_score.macro_score > best_macro + 1e-9:
                    best_macro = win_score.macro_score
                    improved = True
                    champion = winner

            stale = 0 if improved else stale + 1
            generations.append(
                GenerationRecord(
                    generation=gen,
                    parent_bank_digest=parent.bank_digest,
                    candidate_ids=tuple(candidate_ids),
                    selection=selection,
                    diagnosis=diagnosis,
                    champion_id=champion.candidate_id if champion else None,
                    champion_bank_digest=champion.bank_digest if champion else None,
                )
            )
            if improved and winner is not None and self.seed_mode != "empty_each_round":
                parent = load_bank(winner.bank_dir)
            if stale >= self.budgets.plateau_patience:
                stopped_reason = "PLATEAU"
                break

        return EvolveResult(
            champion=champion,
            generations=tuple(generations),
            stopped_reason=stopped_reason,
            archive_root=str(self.archive.root),
            evaluated_candidates=self._evaluated,
        )

    # ------------------------------------------------------------------
    def _propose_materialize(self, packet: EvolutionPacket, parent: LoadedBank) -> MaterializedCandidate | None:
        last_err: BaseException | None = None
        for _repair in range(self.budgets.max_candidate_repairs + 1):
            try:
                patch: CandidatePatch = self.proposer.propose(packet)
            except CandidateInvalid as e:
                last_err = e
                continue
            try:
                mat = materialize_candidate(
                    patch,
                    parent_bank_root=parent.root,
                    parent_bank_digest=parent.bank_digest,
                    out_root=self.candidates_root,
                )
            except CandidateInvalid as e:
                last_err = e
                continue
            report = validate_materialized(mat.bank_dir, known_task_ids=self.known_task_ids)
            if not report.ok:
                last_err = CandidateInvalid(f"validation failed: {report.problems}")
                continue
            return mat
        self._history.append(
            {"error": f"candidate failed after repairs: {last_err!r}", "generation": packet.generation}
        )
        return None

    def _score_candidate(self, mat: MaterializedCandidate) -> CandidateScore:
        bank = load_bank(mat.bank_dir)
        result = self.panel_runner(bank)
        self._evaluated += 1
        scores = {d: dict(t) for d, t in dict(result["scores"]).items()}
        per_domain_means = {
            d: (sum(t.values()) / len(t) if t else 0.0) for d, t in scores.items()
        }
        macro = sum(per_domain_means.values()) / len(per_domain_means) if per_domain_means else 0.0
        cost_raw = dict(result.get("cost") or {})
        cost = CostVector(
            active_components=len(bank.components),
            extra_prompt_tokens=float(cost_raw.get("aux_input_tokens", 0)),
            extra_llm_calls=float(cost_raw.get("extra_llm_calls", 0)),
            extra_tool_calls=float(cost_raw.get("extra_tool_calls", 0)),
            latency_ms=float(cost_raw.get("wall_ms", 0)),
        )
        return CandidateScore(
            candidate_id=mat.candidate_id,
            macro_score=macro,
            per_domain=per_domain_means,
            cost=cost,
            protocol_ok=bool(result.get("protocol_ok", True)),
            budget_ok=bool(result.get("budget_ok", True)),
        )

    def _as_candidate_score(self, bank: LoadedBank, result: Mapping[str, Any]) -> CandidateScore:
        scores = {d: dict(t) for d, t in dict(result["scores"]).items()}
        per_domain_means = {
            d: (sum(t.values()) / len(t) if t else 0.0) for d, t in scores.items()
        }
        macro = sum(per_domain_means.values()) / len(per_domain_means) if per_domain_means else 0.0
        return CandidateScore(
            candidate_id="baseline:" + bank.bank_digest[:12],
            macro_score=macro,
            per_domain=per_domain_means,
            cost=CostVector(),
            protocol_ok=True,
            budget_ok=True,
        )
