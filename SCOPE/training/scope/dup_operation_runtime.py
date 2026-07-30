"""Closed-loop duplicate operation inference + ActionRealizer integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.dup_decision_point import (
    build_decision_points,
    is_evidence_admission_action,
)
from harness.capability.dup_operation import DupOperation
from harness.capability.state import DecisionState
from harness.shadow.action_realizer import ActionRealizer
from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.train_rl import CurateTool
from harness.trajectory import Action
from training.scope.decision_config import DupDecisionConfig, DEFAULT_DECISION_CONFIG
from training.scope.dup_telemetry import AdmissionEvent, DupTelemetryAggregator
from training.scope.operation_scorer import score_operations


@dataclass
class DupOperationRuntimeConfig:
    enabled: bool = True
    record_shadow: bool = True
    fail_on_unsupported: bool = True
    decision_config: DupDecisionConfig = DEFAULT_DECISION_CONFIG
    checkpoint: str = ""
    seed: int = 0


class DupOperationRuntime:
    """Intercept curate actions → score KEEP/SKIP → ActionRealizer → runtime."""

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        *,
        device: torch.device | None = None,
        config: DupOperationRuntimeConfig | None = None,
        vllm_scorer: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if device is None and model is not None:
            self.device = next(model.parameters()).device
        self.config = config or DupOperationRuntimeConfig()
        self.realizer = ActionRealizer()
        self.shadow = DupBilateralShadow()
        self.telemetry = DupTelemetryAggregator()
        self._curate_tool = CurateTool()
        self._vllm_scorer = vllm_scorer

    def supports_typed_operation(self) -> bool:
        return self.config.enabled

    def _state_text(self, state: DecisionState) -> str:
        return state.rendered_context or state.query or ""

    def score_and_predict(
        self, state: DecisionState
    ) -> tuple[DupOperation, dict[str, float]]:
        if self._vllm_scorer is not None:
            result = self._vllm_scorer.score(self._state_text(state))
            return result.predicted, dict(result.scores)
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("DupOperationRuntime requires model or vllm_scorer")
        result = score_operations(
            self.model, self.tokenizer, self._state_text(state), device=self.device
        )
        sk = result.scores[DupOperation.KEEP_EVIDENCE.value]
        ss = result.scores[DupOperation.SKIP_DUPLICATE.value]
        margin = ss - sk
        predicted = self.config.decision_config.predict_from_margin(margin)
        return predicted, dict(result.scores)

    def realize_action(
        self,
        state: DecisionState,
        student_cap: CapabilityAction,
        action: Action,
        *,
        query_id: str = "",
    ) -> Action:
        """Map student curate through operation scorer + ActionRealizer."""
        if not self.config.enabled:
            if self.config.fail_on_unsupported:
                self.telemetry.hidden_fallback_count += 1
            return action

        if not is_evidence_admission_action(student_cap):
            return action

        points = build_decision_points(state, student_cap)
        if not points:
            return action

        predicted, score_map = self.score_and_predict(state)
        primary_cid = points[0].candidate_evidence_id

        sk = score_map.get(DupOperation.KEEP_EVIDENCE.value, 0.0)
        ss = score_map.get(DupOperation.SKIP_DUPLICATE.value, 0.0)
        margin = ss - sk
        cfg = self.config.decision_config

        cand = self.realizer.realize_operation(
            state,
            predicted,
            candidate_id=primary_cid,
            student_action=student_cap,
        )
        if cand is None:
            self.telemetry.action_realizer_failures += 1
            return action

        realized = cand.action
        add_ids = realized.arguments.get("add_ids") or []
        remove_ids = realized.arguments.get("remove_ids") or []

        if self.config.record_shadow:
            shadow_art = self.shadow.analyze_candidate(
                state, student_cap, points[0]
            )
            shadow_op = str(
                (shadow_art.metadata or {}).get("shadow_operation", "")
            )
            route = shadow_art.mode.value.upper()
            curated_before = set(state.curated_document_ids)
            actually = bool(add_ids) and any(
                str(d) not in curated_before for d in add_ids
            )
            self.telemetry.add(
                AdmissionEvent(
                    candidate_evidence_id=primary_cid,
                    candidate_is_duplicate=bool(
                        (shadow_art.metadata or {}).get("candidate_is_duplicate")
                    ),
                    student_operation=predicted.value,
                    shadow_operation=shadow_op,
                    route=route,
                    realized_runtime_action=realized.to_dict(),
                    actually_curated=actually,
                    query_id=query_id,
                    turn_id=state.turn_id,
                )
            )
            # Extended score telemetry for Round 6 audit
            self.telemetry.score_events.append({
                    "episode_id": state.episode_id,
                    "qid": query_id,
                    "turn": state.turn_id,
                    "checkpoint": self.config.checkpoint,
                    "seed": self.config.seed,
                    "label": shadow_op,
                    "score_keep": sk,
                    "score_skip": ss,
                    "margin_skip_minus_keep": margin,
                    "threshold": cfg.threshold,
                    "decision_bias": cfg.decision_bias,
                    "predicted_operation": predicted.value,
                    "executed_operation": predicted.value,
                })

        return Action(
            tools=[self._curate_tool],
            params=[{"add_ids": add_ids, "remove_ids": remove_ids}],
            sources=["dup_operation_realized"],
        )

    def make_pre_step_hook(self, query_id: str = "") -> Callable[[DecisionState, Action], Action]:
        def hook(state: DecisionState, action: Action) -> Action:
            cap = _action_to_cap(action)
            return self.realize_action(state, cap, action, query_id=query_id)

        return hook


def _action_to_cap(action: Action) -> CapabilityAction:
    from training.chat_decision_driver import _action_to_capability

    return _action_to_capability(action)
