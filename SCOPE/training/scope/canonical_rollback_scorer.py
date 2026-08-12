"""Canonical single-backend rollback operation scorer (Round10 followup A2).

Training still uses HF forward for loss. All offline / frozen-live / closed-loop
*inference decisions* must go through this scorer + ``decide_rollback_operation``.

HF↔vLLM score diffs may be logged as diagnostics only; they must not drive a
second inference decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.capability.rollback_operation import RollbackOperation
from training.scope.checkpoint_candidates import global_to_local_id
from training.scope.decide_rollback_operation import (
    RollbackDecision,
    decide_rollback_operation,
)
from training.scope.rollback_operation_runtime import pick_rollback_checkpoint
from training.scope.vllm_rollback_scorer import RollbackScoreResult, VllmRollbackScorer


@dataclass
class CanonicalScoreBundle:
    scores: dict[str, float]
    decision: RollbackDecision
    pred_operation: str
    pred_checkpoint_global_id: str | None
    pred_checkpoint_local_id: str | None
    fallback_reason: str | None
    scorer_backend: str = "vllm_canonical"


class CanonicalRollbackOperationScorer:
    """Unique inference decision contract for rollback operations.

    Prefer the live runtime vLLM scorer so offline replay, frozen-live replay,
    and closed-loop share one scoring + decide path.
    """

    BACKEND = "vllm_canonical"

    def __init__(
        self,
        scorer: VllmRollbackScorer,
        *,
        threshold: float = 0.0,
        disable_replan: bool = True,
    ) -> None:
        self.scorer = scorer
        self.threshold = float(threshold)
        self.disable_replan = bool(disable_replan)

    def score_final_prompt(self, prompt: str) -> RollbackScoreResult:
        return self.scorer.score_final_prompt(prompt)

    def decide_from_scores(
        self,
        scores: dict[str, float],
        *,
        candidate_checkpoint_id: str | None = None,
    ) -> RollbackDecision:
        return decide_rollback_operation(
            score_continue=float(scores.get(RollbackOperation.CONTINUE.value, -1e9)),
            score_replan=float(scores.get(RollbackOperation.REPLAN.value, -1e9)),
            score_rollback=float(scores.get(RollbackOperation.ROLLBACK_TO.value, -1e9)),
            threshold=self.threshold,
            candidate_checkpoint_id=candidate_checkpoint_id,
            disable_replan=self.disable_replan,
        )

    def decide_row(self, row: dict[str, Any], *, prompt_is_final: bool = True) -> CanonicalScoreBundle:
        candidates = row.get("candidate_list") or []
        ck_meta = [
            {
                "checkpoint_id": c.get("checkpoint_id"),
                "turn_id": c.get("relative_turn", c.get("turn_id", 0)),
                "n_curated": c.get("evidence_count", 0),
                "n_pool": c.get("n_pool", 0),
            }
            for c in candidates
        ]
        text = row["effective_input_text"]
        if prompt_is_final:
            result = self.scorer.score_final_prompt(text)
        else:
            result = self.scorer.score(text, available_checkpoints=ck_meta, prompt_is_final=False)
        ck_pick = pick_rollback_checkpoint(ck_meta, int(row.get("turn", 0)))
        decision = self.decide_from_scores(result.scores, candidate_checkpoint_id=ck_pick)
        local_to_global = {
            c.get("local_checkpoint_id"): c.get("checkpoint_id") for c in candidates
        }
        pred_local = global_to_local_id(decision.checkpoint_id, local_to_global)
        valid_ids = set(local_to_global.values())
        fallback_reason = None
        if decision.predicted_operation.value == "ROLLBACK_TO" and (
            not valid_ids or decision.checkpoint_id not in valid_ids
        ):
            fallback_reason = "invalid_checkpoint_prediction"
        return CanonicalScoreBundle(
            scores=dict(result.scores),
            decision=decision,
            pred_operation=decision.predicted_operation.value,
            pred_checkpoint_global_id=decision.checkpoint_id,
            pred_checkpoint_local_id=pred_local,
            fallback_reason=fallback_reason,
            scorer_backend=self.BACKEND,
        )

    def replay_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        total = len(rows)
        for idx, row in enumerate(rows):
            if idx % 25 == 0 or idx + 1 == total:
                print(f"[canonical-replay] {idx}/{total}", flush=True)
            bundle = self.decide_row(row, prompt_is_final=True)
            out.append(
                {
                    **row,
                    "event_id": row.get("event_id")
                    or f"{row.get('query_id')}:{row.get('turn')}:{idx}",
                    "vllm_logits": dict(bundle.scores),
                    "canonical_logits": dict(bundle.scores),
                    "pred_operation": bundle.pred_operation,
                    "pred_checkpoint_local_id": bundle.pred_checkpoint_local_id,
                    "pred_checkpoint_global_id": bundle.pred_checkpoint_global_id,
                    "fallback_reason": bundle.fallback_reason,
                    "scorer_backend": bundle.scorer_backend,
                    "decision_threshold": self.threshold,
                    "disable_replan": self.disable_replan,
                }
            )
        return out


def decide_from_saved_logits(
    row: dict[str, Any],
    *,
    logits_key: str = "vllm_logits",
    threshold: float = 0.0,
    disable_replan: bool = True,
) -> CanonicalScoreBundle:
    """Re-decide from saved backend logits via the canonical decide contract."""
    scores = row.get(logits_key) or row.get("canonical_logits") or row.get("hf_logits") or {}
    candidates = row.get("candidate_list") or []
    ck_meta = [
        {
            "checkpoint_id": c.get("checkpoint_id"),
            "turn_id": c.get("relative_turn", c.get("turn_id", 0)),
            "n_curated": c.get("evidence_count", 0),
            "n_pool": c.get("n_pool", 0),
        }
        for c in candidates
    ]
    ck_pick = pick_rollback_checkpoint(ck_meta, int(row.get("turn", 0)))
    decision = decide_rollback_operation(
        score_continue=float(scores.get("CONTINUE", -1e9)),
        score_replan=float(scores.get("REPLAN", -1e9)),
        score_rollback=float(scores.get("ROLLBACK_TO", -1e9)),
        threshold=threshold,
        candidate_checkpoint_id=ck_pick,
        disable_replan=disable_replan,
    )
    local_to_global = {
        c.get("local_checkpoint_id"): c.get("checkpoint_id") for c in candidates
    }
    pred_local = global_to_local_id(decision.checkpoint_id, local_to_global)
    valid_ids = set(local_to_global.values())
    fallback_reason = None
    if decision.predicted_operation.value == "ROLLBACK_TO" and (
        not valid_ids or decision.checkpoint_id not in valid_ids
    ):
        fallback_reason = "invalid_checkpoint_prediction"
    return CanonicalScoreBundle(
        scores={k: float(v) for k, v in scores.items()},
        decision=decision,
        pred_operation=decision.predicted_operation.value,
        pred_checkpoint_global_id=decision.checkpoint_id,
        pred_checkpoint_local_id=pred_local,
        fallback_reason=fallback_reason,
        scorer_backend="canonical_from_saved_logits",
    )
