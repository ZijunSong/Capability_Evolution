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
from rsi.evolution.dedup import DedupCache, ast_digest
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
    # 候选被拒/panel 报错的原始记录。9fhn3 复盘：这些只进了内存 _history，
    # 从未落盘，导致 18 次提议全废却查不到一个拒绝原因。必须持久化。
    history: tuple[dict[str, JSONValue], ...] = ()

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
            "history": [h for h in self.history],
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
        analysis_runner: Callable[[LoadedBank], Mapping[str, Any]] | None = None,
    ):
        self.proposer = proposer
        self.archive = archive
        self.panel_runner = panel_runner
        # Refactor guide §9: analysis_runner runs the CURRENT parent on
        # work_train and returns {"evidence": EvidenceStore-like mapping with
        # real action/observation fragments} — separated from panel_runner
        # (scoring on harness_select) so selection cost and analysis cost are
        # distinct and the packet can quote REAL failures.
        self.analysis_runner = analysis_runner
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
        self._dedup_cache = DedupCache()

    def run(self) -> EvolveResult:
        generations: list[GenerationRecord] = []
        # Round seed: empty_each_round starts from the empty bank every round;
        # warm_archive allows old source as parents but still re-measures.
        if self.seed_mode == "empty_each_round" or self.parent_bank_root is None:
            parent: LoadedBank = load_empty_bank()
        else:
            parent = load_bank(self.parent_bank_root)

        champion: MaterializedCandidate | None = None
        # PR-5 (audit §8.1): best_macro starts at the PARENT's measured macro,
        # not None — the first executable candidate must not auto-become the
        # champion merely by existing.  The parent result is re-measured each
        # generation under the same panel; gen-0's initial value comes from
        # the pre-run baseline scores when available, else it is established
        # by gen-0's own parent measurement below.
        best_macro: float | None = None
        if self._baseline_scores:
            per_domain_means = {
                d: (sum(t.values()) / len(t) if t else 0.0)
                for d, t in self._baseline_scores.items()
            }
            if per_domain_means:
                best_macro = sum(per_domain_means.values()) / len(per_domain_means)
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
            analysis = None
            if self.analysis_runner is not None:
                try:
                    analysis = self.analysis_runner(parent)
                except Exception as e:  # noqa: BLE001 - analysis is enrichment:
                    # a failing analysis pass must not kill the search; the
                    # packet falls back to telemetry-only fragments.
                    self._history.append(
                        {"analysis_error": repr(e), "generation": gen}
                    )
                    analysis = None
            packet = self.packet_builder(
                generation=gen,
                policy_digest=str(parent_result.get("policy_digest", "")),
                parent_bank=parent,
                baseline_scores=self._baseline_scores or parent_scores,
                parent_scores=parent_scores,
                diagnosis=diagnosis,
                failure_fragments=self._fragments_from_per_task(
                    parent_result.get("per_task") or {}, failure=True
                )
                + self._fragments_from_analysis(analysis, failure=True),
                success_summaries=self._fragments_from_per_task(
                    parent_result.get("per_task") or {}, failure=False
                )
                + self._fragments_from_analysis(analysis, failure=False),
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
                # PR-03 (guide §4.4): parent replacement is decided by the
                # CURRENT generation's paired result — candidate vs the
                # parent measured THIS generation on the same panel.  The
                # old rule (`win > best_macro`) let one noisy historical
                # high-water mark block a genuinely improved current parent
                # indefinitely, and conversely treated a historical max as
                # reliable teacher evidence.
                if win_score.macro_score > baseline_score.macro_score + 1e-9:
                    improved = True
                    champion = winner
                    # historical_best_macro is display-only telemetry now
                    best_macro = max(best_macro or 0.0, win_score.macro_score)

            stale = 0 if improved else stale + 1
            # PR-7 (audit §8.2): repeated equivalent proposals are a DISTINCT
            # stop condition from "no quality gain" — the proposer is stalled
            # on the same idea, not the search space being exhausted.  Track
            # dup streaks and name them honestly.
            dup_ids = [h for h in self._history[-len(candidate_ids or ()) :] if h.get("dedup")]
            if candidate_ids and len(dup_ids) == len(candidate_ids):
                self._dup_streak = getattr(self, "_dup_streak", 0) + 1
            else:
                self._dup_streak = 0
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
            if improved and winner is not None:
                # Within-run generations COMPOSE on the champion so banks grow
                # into multi-component harnesses (task book §7.2 parent
                # selection: keep the champion as parent).  ``empty_each_round``
                # governs ROUND starts, not generation starts.
                parent = load_bank(winner.bank_dir)
            if self._dup_streak >= self.budgets.plateau_patience:
                stopped_reason = "PROPOSAL_STALLED"
                break
            if stale >= self.budgets.plateau_patience:
                stopped_reason = "PLATEAU"
                break

        return EvolveResult(
            champion=champion,
            generations=tuple(generations),
            stopped_reason=stopped_reason,
            archive_root=str(self.archive.root),
            evaluated_candidates=self._evaluated,
            history=tuple(self._history),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fragments_from_per_task(
        per_task: Mapping[str, Any], *, failure: bool, limit: int = 6, max_chars: int = 400
    ) -> tuple[TraceFragment, ...]:
        """PR-4 (audit §5.1): the packet's failure_fragments/success_summaries
        were never filled — the proposer saw only labels.  Build bounded text
        fragments from the real per-task telemetry (decisions, wall, metric,
        component errors) so hypotheses can quote concrete evidence.

        PR-03 (guide §4.1): outcome routing uses the unified classifier —
        ``success`` cases are strict successes only; AppWorld partial credit
        (value>0, success=False) is a PARTIAL_FAILURE fragment.  INVALID
        measurements never enter either bucket."""
        from rsi.evolution.outcomes import Outcome, classify_outcome

        fragments: list[TraceFragment] = []
        for tid in sorted(per_task):
            meta = dict(per_task[tid] or {})
            try:
                outcome = classify_outcome(meta)
            except ValueError:
                outcome = Outcome.INVALID
            if outcome is Outcome.INVALID:
                continue
            is_failure = outcome in (Outcome.PARTIAL_FAILURE, Outcome.FAILURE)
            if failure != is_failure:
                continue
            comp = meta.get("component_errors") or {}
            text = (
                f"value={meta.get('value')} success={meta.get('success')} "
                f"outcome={outcome.value} terminated={meta.get('terminated_reason')} "
                f"n_decisions={meta.get('n_decisions', 0)} wall_ms={meta.get('wall_ms', 0):.0f} "
                f"metric={meta.get('metric_name')}"
                + (f" component_errors={comp}" if comp else "")
            )[:max_chars]
            fragments.append(
                TraceFragment(
                    task_id=tid,
                    outcome="fail" if failure else "success",
                    events_text=text,
                    decisions=int(meta.get("n_decisions", 0) or 0),
                    error_kind=(next(iter(comp), None) if comp else None),
                )
            )
            if len(fragments) >= limit:
                break
        return tuple(fragments)

    @staticmethod
    def _fragments_from_analysis(
        analysis: Mapping[str, Any] | None, *, failure: bool, limit: int = 6, max_chars: int = 500
    ) -> tuple[TraceFragment, ...]:
        """Real action/observation fragments from the analysis runner's
        EvidenceStore (guide §9): telemetry alone cannot distinguish a wrong
        API choice from a missed constraint — the packet carries the actual
        steps.  Only work_train/evolution_train evidence ever reaches here.

        PR-04 (guide §4.3): classified FAILURE WINDOWS take priority over the
        raw case fragments — they pair the failing action with its
        predecessor, tag the mechanism (invalid_action | api_error |
        no_output | budget), and carry event refs for
        ``failure_evidence_refs``."""
        if not analysis:
            return ()
        if failure and analysis.get("failure_windows"):
            out: list[TraceFragment] = []
            for w in list(analysis["failure_windows"])[:limit]:
                out.append(
                    TraceFragment(
                        task_id=str(w.get("task_id", "?")),
                        outcome="fail",
                        events_text=str(w.get("evidence_text", ""))[:max_chars],
                        decisions=int(w.get("n_decisions", 0) or 0),
                        error_kind=str(w.get("failure_type", "unknown")),
                    )
                )
            return tuple(out)
        cases = (analysis.get("failure_cases") if failure else analysis.get("success_cases")) or []
        out = []
        for c in list(cases)[:limit]:
            text = str(c.get("fragment", ""))[:max_chars]
            if not text:
                continue
            out.append(
                TraceFragment(
                    task_id=str(c.get("task_id", "?")),
                    outcome="fail" if failure else "success",
                    events_text=text,
                    decisions=int(c.get("n_decisions", 0) or 0),
                    error_kind=c.get("error_kind"),
                )
            )
        return tuple(out)

    def _propose_materialize(self, packet: EvolutionPacket, parent: LoadedBank) -> MaterializedCandidate | None:
        last_err: BaseException | None = None
        retry_notes: list[str] = []
        for _repair in range(self.budgets.max_candidate_repairs + 1):
            try:
                patch: CandidatePatch = self.proposer.propose(packet, retry_notes=tuple(retry_notes))
            except CandidateInvalid as e:
                last_err = e
                retry_notes.append(f"proposer output rejected: {e}")
                # 实时落 stdout：否则拒绝原因只活在内存里，跑完查无可查（9fhn3 教训）
                print(f"[proposer] gen{packet.generation} attempt{_repair} rejected: {e}", flush=True)
                continue
            # PR-04 (guide §4.3): the evidence contract — a hypothesis that
            # only says "be more careful" without citing failure evidence is
            # rejected before the expensive materialization.
            from rsi.evolution.failure_windows import hypothesis_quality_problems

            hq = hypothesis_quality_problems(
                getattr(patch, "hypothesis", ""), tuple(getattr(patch, "evidence_refs", ()) or ())
            )
            if hq:
                last_err = CandidateInvalid("evidence contract violated: " + "; ".join(hq))
                retry_notes.append("evidence contract violated: " + "; ".join(hq))
                print(f"[proposer] gen{packet.generation} attempt{_repair} evidence-rejected: {hq}", flush=True)
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
                retry_notes.append(f"candidate rejected at materialize/load: {e}")
                print(f"[proposer] gen{packet.generation} attempt{_repair} materialize rejected: {e}", flush=True)
                continue
            report = validate_materialized(mat.bank_dir, known_task_ids=self.known_task_ids)
            if not report.ok:
                last_err = CandidateInvalid(f"validation failed: {report.problems}")
                retry_notes.append("validation failed: " + "; ".join(report.problems[:8]))
                print(f"[proposer] gen{packet.generation} attempt{_repair} validation rejected: "
                      f"{str(report.problems[:4])}", flush=True)
                continue
            print(f"[proposer] gen{packet.generation} ACCEPTED candidate {mat.candidate_id}", flush=True)
            return mat
        self._history.append(
            {"error": f"candidate failed after repairs: {last_err!r}", "generation": packet.generation}
        )
        print(f"[proposer] gen{packet.generation} gave up after repairs: {last_err!r}", flush=True)
        return None

    def _score_candidate(self, mat: MaterializedCandidate) -> CandidateScore:
        bank = load_bank(mat.bank_dir)
        # PR-7 (audit §8.2): content-addressed dedup BEFORE the expensive
        # panel.  gen1/2/3 re-proposals of the same component used to consume
        # one full panel each; exact duplicates now reuse the prior result
        # and AST near-duplicates are flagged in the history.
        bank_digest = bank.bank_digest
        prior = self._dedup_cache.lookup_exact(bank_digest)
        if prior is not None:
            self._history.append(
                {
                    "candidate_id": mat.candidate_id,
                    "generation": mat.__dict__.get("generation"),
                    "dedup": "exact_dup",
                    "of": prior.get("candidate_id", bank_digest[:12]),
                    "macro_score": prior.get("macro_score"),
                }
            )
            return CandidateScore(
                candidate_id=mat.candidate_id,
                macro_score=float(prior.get("macro_score", 0.0)),
                per_domain={d: dict(t) for d, t in (prior.get("per_domain") or {}).items()},
                cost=CostVector(
                    active_components=len(bank.components),
                    extra_prompt_tokens=float((prior.get("cost") or {}).get("aux_input_tokens", 0)),
                    extra_llm_calls=float((prior.get("cost") or {}).get("extra_llm_calls", 0)),
                    extra_tool_calls=float((prior.get("cost") or {}).get("extra_tool_calls", 0)),
                    latency_ms=float((prior.get("cost") or {}).get("wall_ms_total", 0)),
                ),
                protocol_ok=bool(prior.get("protocol_ok", True)),
                budget_ok=bool(prior.get("budget_ok", True)),
                error_rate=float(prior.get("error_rate", 0.0)),
            )
        try:
            result = self.panel_runner(bank)
        except Exception as e:  # noqa: BLE001 - a broken candidate must not kill the search
            self._history.append(
                {"candidate_id": mat.candidate_id, "error": f"panel failed: {e!r}"}
            )
            self._evaluated += 1
            return CandidateScore(
                candidate_id=mat.candidate_id,
                macro_score=0.0,
                per_domain={},
                cost=CostVector(active_components=len(bank.components)),
                protocol_ok=False,
                budget_ok=True,
            )
        self._evaluated += 1
        comp_errors = dict(result.get("component_errors") or {})
        if comp_errors or not result.get("protocol_ok", True):
            self._history.append(
                {
                    "candidate_id": mat.candidate_id,
                    "generation_error": "candidate marked protocol-failed: "
                    + (str(comp_errors)[:400] if comp_errors else "protocol_ok=false"),
                }
            )
        scores = {d: dict(t) for d, t in dict(result["scores"]).items()}
        per_domain_means = {
            d: (sum(t.values()) / len(t) if t else 0.0) for d, t in scores.items()
        }
        macro = sum(per_domain_means.values()) / len(per_domain_means) if per_domain_means else 0.0
        # record for future dedup (content + AST fingerprints); a near-dup
        # hit is surfaced in history so the proposer feedback can name the
        # earlier equivalent candidate (PROPOSAL_STALLED evidence).
        ast_digests = {}
        for comp in bank.components:
            if comp.manifest.files:
                fp = Path(bank.root) / comp.manifest.files[0]
                if fp.is_file():
                    ast_digests[comp.manifest.id] = ast_digest(fp.read_text("utf-8"))
        near_of = self._dedup_cache.record(
            bank_digest=bank_digest,
            ast_digests=ast_digests,
            result={"macro_score": 0.0},
        )
        if near_of:
            self._history.append(
                {
                    "candidate_id": mat.candidate_id,
                    "dedup": "near_dup",
                    "of": near_of,
                }
            )
        cost_raw = dict(result.get("cost") or {})
        cost = CostVector(
            active_components=len(bank.components),
            extra_prompt_tokens=float(cost_raw.get("aux_input_tokens", 0)),
            # CostLedger emits wall_ms_total (and no extra_llm_calls counter);
            # read the canonical keys and fall back to the legacy alias so a
            # partially-wired ledger cannot silently score latency as zero.
            extra_llm_calls=float(cost_raw.get("extra_llm_calls", 0)),
            extra_tool_calls=float(cost_raw.get("extra_tool_calls", 0)),
            latency_ms=float(cost_raw.get("wall_ms_total", cost_raw.get("wall_ms", 0))),
        )
        return CandidateScore(
            candidate_id=mat.candidate_id,
            macro_score=macro,
            per_domain=per_domain_means,
            cost=cost,
            protocol_ok=bool(result.get("protocol_ok", True)),
            budget_ok=bool(result.get("budget_ok", True)),
            error_rate=float(result.get("error_rate", 0.0)),
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
