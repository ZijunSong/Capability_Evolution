"""Closed-loop rollback typed operation inference + hard executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from harness.capability.rollback_operation import RollbackOperation
from harness.capability.state import DecisionState
from harness.recovery.checkpoint_store import CheckpointStore
from harness.recovery.recovery_budget import RecoveryBudget
from harness.recovery.rollback_runtime import RollbackRuntime
from harness.recovery.stagnation_detector import StagnationDetector
from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope.rollback_operation_objectives import score_rollback_operations
from training.scope.rollback_action_realizer import RollbackActionRealizer
from training.scope.rollback_decision_state import build_rollback_decision_state
from training.scope.rollback_shadow import RollbackBilateralShadow
from training.scope.vllm_rollback_scorer import VllmRollbackScorer


@dataclass
class RollbackTelemetryEvent:
    query_id: str
    turn_id: int
    student_operation: str
    shadow_operation: str
    shadow_checkpoint_id: str | None
    predicted_checkpoint_id: str | None
    route: str
    rollback_success: bool = False
    state_hash_restore: bool = True


@dataclass
class RollbackOperationRuntimeConfig:
    enabled: bool = True
    threshold: float = 0.0
    soft_replan_only: bool = False
    hint: str = ""
    checkpoint_label: str = ""
    # Round10 / followup contract: closed-loop must share disable_replan with
    # offline + frozen-live canonical inference (see CanonicalRollbackOperationScorer).
    disable_replan: bool = True


def pick_rollback_checkpoint(
    available_checkpoints: list[dict[str, Any]],
    current_turn_id: int,
    suggested: str | None = None,
) -> str | None:
    if suggested:
        ids = {str(c.get("checkpoint_id", "")) for c in available_checkpoints}
        if suggested in ids:
            return suggested
    eligible = [
        c
        for c in available_checkpoints
        if int(c.get("turn_id", -1)) < int(current_turn_id)
    ]
    if not eligible:
        eligible = list(available_checkpoints)
    if not eligible:
        return None
    best = max(eligible, key=lambda c: int(c.get("turn_id", 0)))
    return str(best.get("checkpoint_id")) or None


class RollbackOperationRuntime:
    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        *,
        device: torch.device | None = None,
        config: RollbackOperationRuntimeConfig | None = None,
        vllm_scorer: VllmRollbackScorer | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if device is None and model is not None:
            self.device = next(model.parameters()).device
        self.config = config or RollbackOperationRuntimeConfig()
        self._vllm_scorer = vllm_scorer
        self.shadow = RollbackBilateralShadow()
        self.realizer = RollbackActionRealizer()
        self.detector = StagnationDetector()
        self.events: list[RollbackTelemetryEvent] = []
        self.invalid_checkpoint_predictions = 0
        self.budget_violations = 0
        self.hidden_fallbacks = 0

    def fork_for_query(
        self,
        env: Any,
        *,
        query_id: str,
        max_turns: int,
    ) -> _QueryRollbackContext:
        store = CheckpointStore(branch_id=query_id)
        budget = RecoveryBudget(max_rollbacks=3)
        runtime = RollbackRuntime(store, budget)
        return _QueryRollbackContext(
            parent=self,
            env=env,
            query_id=query_id,
            max_turns=max_turns,
            store=store,
            rollback_runtime=runtime,
        )


@dataclass
class _QueryRollbackContext:
    parent: RollbackOperationRuntime
    env: Any
    query_id: str
    max_turns: int
    store: CheckpointStore
    rollback_runtime: RollbackRuntime
    events: list[RollbackTelemetryEvent] = field(default_factory=list)

    def _state_text(self, ds: dict[str, Any]) -> str:
        return str(ds.get("rendered_context") or ds.get("student_state_text") or "")

    def score_and_decide(
        self,
        ds: dict[str, Any],
        *,
        suggested_checkpoint_id: str | None = None,
    ) -> tuple[RollbackOperation, str | None, dict[str, float]]:
        text = self._state_text(ds)
        ck_meta = list(ds.get("available_checkpoints") or [])
        if self.parent._vllm_scorer is not None:
            result = self.parent._vllm_scorer.score(
                text, available_checkpoints=ck_meta
            )
            s_cont = result.scores[RollbackOperation.CONTINUE.value]
            s_replan = result.scores[RollbackOperation.REPLAN.value]
            s_roll = result.scores[RollbackOperation.ROLLBACK_TO.value]
            predicted = result.predicted
            score_map = dict(result.scores)
        else:
            if self.parent.model is None or self.parent.tokenizer is None:
                raise RuntimeError("RollbackOperationRuntime needs model or vllm_scorer")
            s_cont, s_replan, s_roll = score_rollback_operations(
                self.parent.model,
                self.parent.tokenizer,
                text,
                device=self.parent.device,
                available_checkpoints=ck_meta,
                hint=self.parent.config.hint,
            )
            score_map = {
                RollbackOperation.CONTINUE.value: float(s_cont.detach().item()),
                RollbackOperation.REPLAN.value: float(s_replan.detach().item()),
                RollbackOperation.ROLLBACK_TO.value: float(s_roll.detach().item()),
            }
            predicted = max(
                score_map,
                key=score_map.get,
            )
            predicted = RollbackOperation(predicted)

        ck_pick = pick_rollback_checkpoint(
            ck_meta,
            int(ds.get("turn_id", 0)),
            suggested=suggested_checkpoint_id,
        )
        decision = decide_rollback_operation(
            score_continue=score_map[RollbackOperation.CONTINUE.value],
            score_replan=score_map[RollbackOperation.REPLAN.value],
            score_rollback=score_map[RollbackOperation.ROLLBACK_TO.value],
            threshold=self.parent.config.threshold,
            candidate_checkpoint_id=ck_pick,
            disable_replan=bool(self.parent.config.disable_replan),
        )
        op = decision.predicted_operation
        if self.parent.config.soft_replan_only and op == RollbackOperation.ROLLBACK_TO:
            op = RollbackOperation.REPLAN
        return op, decision.checkpoint_id, score_map

    def make_pre_step_hook(self) -> Callable[[DecisionState, Any], Any]:
        cfg = self.parent.config

        def hook(state: DecisionState, action: Any) -> Any:
            if not cfg.enabled:
                return action
            self.store.save_from_env(self.env, turn_id=int(state.turn_id))
            failure = self.parent.detector.observe_turn(
                self.env, checkpoint_store=self.store
            )
            suggested = failure.suggested_checkpoint_id if failure else None
            if failure is None:
                from harness.capability.rollback_operation import RollbackReasonCode
                from harness.recovery.stagnation_detector import FailureEvent as FE

                shadow_label = self.parent.shadow.label_failure_event(
                    FE(RollbackReasonCode.NONE, "healthy"),
                    healthy_continue=True,
                )
            else:
                shadow_label = self.parent.shadow.label_failure_event(failure)

            ds = build_rollback_decision_state(
                state,
                recent_queries=list(self.env.wm.search_history),
                available_checkpoints=self.store.lightweight_metadata(),
                remaining_search_budget=max(
                    0, self.max_turns - int(self.env._current_turn)
                ),
                remaining_recovery_budget=self.rollback_runtime.budget.remaining(),
                branch_id=self.query_id,
                state_hash=self.env.wm.snapshot_hash(),
            )
            pred_op, pred_ck, _ = self.score_and_decide(
                ds, suggested_checkpoint_id=suggested
            )
            valid_ids = {
                str(c.get("checkpoint_id", ""))
                for c in ds.get("available_checkpoints") or []
            }
            if pred_op == RollbackOperation.ROLLBACK_TO and pred_ck not in valid_ids:
                self.parent.invalid_checkpoint_predictions += 1
                pred_op = RollbackOperation.REPLAN
                pred_ck = None
            if pred_op == RollbackOperation.ROLLBACK_TO and not self.rollback_runtime.budget.can_rollback():
                self.parent.budget_violations += 1
                pred_op = RollbackOperation.REPLAN
                pred_ck = None

            from training.scope.decide_rollback_operation import RollbackDecision

            decision = RollbackDecision(
                predicted_operation=pred_op,
                checkpoint_id=pred_ck if pred_op == RollbackOperation.ROLLBACK_TO else None,
            )
            ok = self.parent.realizer.realize(
                self.env, decision, self.rollback_runtime
            )
            if not ok and pred_op == RollbackOperation.ROLLBACK_TO:
                self.parent.budget_violations += 1

            tel = (
                self.rollback_runtime.telemetry[-1]
                if self.rollback_runtime.telemetry
                else None
            )
            restore_ok = True
            if tel and tel.operation == RollbackOperation.ROLLBACK_TO.value:
                restore_ok = tel.success

            ev = RollbackTelemetryEvent(
                query_id=self.query_id,
                turn_id=int(state.turn_id),
                student_operation=pred_op.value,
                shadow_operation=shadow_label.operation.value,
                shadow_checkpoint_id=shadow_label.checkpoint_id,
                predicted_checkpoint_id=pred_ck,
                route=shadow_label.route,
                rollback_success=bool(tel and tel.success),
                state_hash_restore=restore_ok,
            )
            self.events.append(ev)
            self.parent.events.append(ev)
            return action

        return hook
