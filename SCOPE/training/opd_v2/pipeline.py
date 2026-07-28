"""Build OPDTransitionV2 from DecisionState via shadow modules."""

from __future__ import annotations

from typing import Any

from harness.artifacts.schema import GuidanceMode
from harness.artifacts.visibility import mask_artifact_if_invalid
from harness.capability.action_space import CapabilityAction
from harness.capability.adapters import render_capability_action
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import DecisionState
from harness.shadow.registry import ShadowRegistry, build_default_registry
from training.opd_v2.candidates import fill_recommended_action
from training.opd_v2.router import GuidanceRouter
from training.opd_v2.transitions import OPDTransitionV2, config_hash_from_dict


def build_transitions_for_step(
    state: DecisionState,
    student_action: CapabilityAction,
    *,
    registry: ShadowRegistry | None = None,
    selector: RuleBasedCriticalStateSelector | None = None,
    final_reward: float = 0.0,
    policy_version: str = "",
    tokenizer_version: str = "",
    config: dict[str, Any] | None = None,
    teacher_state_text: str | None = None,
) -> list[OPDTransitionV2]:
    """Run selector → shadow → visibility → router → transitions (no env mutation)."""
    registry = registry or build_default_registry()
    selector = selector or RuleBasedCriticalStateSelector(SelectorConfig())
    router = GuidanceRouter()
    cfg = config or {}
    cfg_hash = config_hash_from_dict(cfg)

    module_ids = selector.select(state, student_action)
    transitions: list[OPDTransitionV2] = []
    student_action_text = render_capability_action(student_action)
    student_state_text = state.rendered_context or state.query

    for mid in module_ids:
        if not registry.has(mid):
            continue
        module = registry.get(mid)
        artifact = module.analyze(state, student_action)
        if artifact.mode == GuidanceMode.CORRECT:
            artifact = fill_recommended_action(state, artifact)
        artifact, _vis = mask_artifact_if_invalid(state, artifact)
        decision = router.route(state, artifact, module=module)

        rec_text = None
        if decision.artifact.recommended_action is not None:
            rec_text = render_capability_action(decision.artifact.recommended_action)

        # Teacher-conditioned prefix for endorse: student state + compact artifact
        t_state = teacher_state_text
        if t_state is None and decision.mode == GuidanceMode.ENDORSE:
            t_state = (
                student_state_text
                + f"\n\n[PRIVILEGED {mid}] reason={decision.artifact.reason_code}"
            )

        validity = 1 if decision.mode != GuidanceMode.IGNORE and decision.validation.valid else 0
        if decision.mode == GuidanceMode.IGNORE:
            continue

        transitions.append(
            OPDTransitionV2.build(
                episode_id=state.episode_id,
                task_id=state.task_id,
                turn_id=state.turn_id,
                module_id=mid,
                mode=decision.mode,
                reason_code=decision.artifact.reason_code,
                student_state_text=student_state_text,
                student_action_text=student_action_text,
                artifact=decision.artifact,
                teacher_state_text=t_state,
                recommended_action_text=rec_text,
                validity_mask=validity,
                teacher_confidence=float(decision.artifact.confidence),
                final_reward=final_reward,
                module_weight=1.0,
                policy_version=policy_version,
                tokenizer_version=tokenizer_version,
                wm_snapshot_hash=state.wm_snapshot_hash,
                state_hash=state.state_hash(),
                config_hash=cfg_hash,
                metadata={
                    "triggers": [
                        {"module_id": t.module_id, "trigger": t.trigger}
                        for t in selector.last_triggers
                        if t.module_id == mid
                    ],
                    "validation_reasons": list(decision.validation.reasons),
                },
            )
        )
    return transitions
