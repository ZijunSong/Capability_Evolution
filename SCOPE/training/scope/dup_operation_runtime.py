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
from training.scope.decide_dup_operation import decide_dup_operation
from training.scope.decision_config import DupDecisionConfig, DEFAULT_DECISION_CONFIG
from training.scope.dup_telemetry import AdmissionEvent, DupTelemetryAggregator
from training.scope.live_dup_decision_trace import (
    LiveDupDecisionTraceWriter,
    make_trace_from_decision,
    sha256_text,
)
from training.scope.operation_scorer import score_operations
from training.scope.prompting import format_operation_prompt


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
        self._trace_writer: LiveDupDecisionTraceWriter | None = None
        self._decision_index = 0

    def set_trace_writer(self, writer: LiveDupDecisionTraceWriter | None) -> None:
        self._trace_writer = writer

    def fork_for_query(self) -> DupOperationRuntime:
        """Per-query runtime sharing scorer; isolated telemetry and decision index."""
        rt = DupOperationRuntime(
            self.model,
            self.tokenizer,
            device=self.device,
            config=self.config,
            vllm_scorer=self._vllm_scorer,
        )
        rt.set_trace_writer(self._trace_writer)
        return rt

    def supports_typed_operation(self) -> bool:
        return self.config.enabled

    def _state_text(self, state: DecisionState) -> str:
        return state.rendered_context or state.query or ""

    def score_and_predict(
        self,
        state: DecisionState,
        *,
        candidate_id: str | None = None,
        curated_document_ids: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[DupOperation, dict[str, float]]:
        curated = curated_document_ids if curated_document_ids is not None else list(
            state.curated_document_ids
        )
        state_text = self._state_text(state)
        if self._vllm_scorer is not None:
            result = self._vllm_scorer.score(
                state_text,
                candidate_id=candidate_id,
                curated_document_ids=curated,
            )
            return result.predicted, dict(result.scores)
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("DupOperationRuntime requires model or vllm_scorer")
        result = score_operations(
            self.model,
            self.tokenizer,
            state_text,
            device=self.device,
            candidate_id=candidate_id,
            curated_document_ids=curated,
        )
        sk = result.scores[DupOperation.KEEP_EVIDENCE.value]
        ss = result.scores[DupOperation.SKIP_DUPLICATE.value]
        predicted = decide_dup_operation(
            score_keep=sk,
            score_skip=ss,
            threshold=self.config.decision_config.effective_threshold(),
        ).predicted_operation
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

        primary_cid = points[0].candidate_evidence_id
        predicted, score_map = self.score_and_predict(
            state,
            candidate_id=primary_cid,
            curated_document_ids=list(state.curated_document_ids),
        )

        sk = score_map.get(DupOperation.KEEP_EVIDENCE.value, 0.0)
        ss = score_map.get(DupOperation.SKIP_DUPLICATE.value, 0.0)
        decision = decide_dup_operation(
            score_keep=sk,
            score_skip=ss,
            threshold=self.config.decision_config.effective_threshold(),
        )
        margin = decision.margin
        cfg = self.config.decision_config
        predicted_pre = predicted

        cand = self.realizer.realize_operation(
            state,
            predicted,
            candidate_id=primary_cid,
            student_action=student_cap,
        )
        if cand is None:
            self.telemetry.action_realizer_failures += 1
            if self._trace_writer is not None:
                state_dict = state.to_dict() if hasattr(state, "to_dict") else {}
                prompt = format_operation_prompt(
                    self._state_text(state),
                    candidate_id=primary_cid,
                    curated_document_ids=list(state.curated_document_ids),
                )
                trace = make_trace_from_decision(
                    query_id=query_id,
                    turn_index=state.turn_id,
                    decision_index=self._decision_index,
                    decision_state=state_dict,
                    rendered_prompt=prompt,
                    input_ids=[],
                    score_keep=sk,
                    score_skip=ss,
                    threshold=cfg.effective_threshold(),
                    threshold_source="per_seed" if cfg.threshold != 0.0 else "fixed_zero",
                    threshold_key=f"seed{self.config.seed}",
                    predicted_pre=predicted_pre,
                    predicted_post=predicted_pre,
                    candidate_evidence_id=primary_cid,
                    shadow_label="",
                    shadow_route="",
                    actually_curated=False,
                    action_payload={},
                    model_id=self.config.checkpoint,
                    checkpoint_path=self.config.checkpoint,
                    checkpoint_sha256=sha256_text(self.config.checkpoint),
                    seed=self.config.seed,
                    backend="vllm_live" if self._vllm_scorer else "hf_replay",
                    fallback_used=True,
                    fallback_reason="action_realizer_failure",
                )
                self._trace_writer.write(trace)
                self._decision_index += 1
            return action

        realized = cand.action
        add_ids = realized.arguments.get("add_ids") or []
        remove_ids = realized.arguments.get("remove_ids") or []
        predicted_post = predicted_pre

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
            if self._trace_writer is not None:
                state_dict = state.to_dict() if hasattr(state, "to_dict") else {}
                prompt = format_operation_prompt(
                    self._state_text(state),
                    candidate_id=primary_cid,
                    curated_document_ids=list(state.curated_document_ids),
                )
                input_ids: list[int] = []
                if self.tokenizer is not None:
                    input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                trace = make_trace_from_decision(
                    query_id=query_id,
                    turn_index=state.turn_id,
                    decision_index=self._decision_index,
                    decision_state=state_dict,
                    rendered_prompt=prompt,
                    input_ids=input_ids,
                    score_keep=sk,
                    score_skip=ss,
                    threshold=cfg.effective_threshold(),
                    threshold_source="per_seed" if cfg.threshold != 0.0 else "fixed_zero",
                    threshold_key=f"seed{self.config.seed}",
                    predicted_pre=predicted_pre,
                    predicted_post=predicted_post,
                    candidate_evidence_id=primary_cid,
                    shadow_label=shadow_op,
                    shadow_route=route,
                    actually_curated=actually,
                    action_payload=realized.to_dict(),
                    model_id=self.config.checkpoint,
                    checkpoint_path=self.config.checkpoint,
                    checkpoint_sha256=sha256_text(self.config.checkpoint),
                    seed=self.config.seed,
                    backend="vllm_live" if self._vllm_scorer else "hf_replay",
                )
                self._trace_writer.write(trace)
                self._decision_index += 1

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
