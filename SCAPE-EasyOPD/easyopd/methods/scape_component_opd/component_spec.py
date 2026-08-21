from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import EffectType, LossMode, Realizability

ComponentFn = Callable[..., Any]


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _true(*_args: Any, **_kwargs: Any) -> bool:
    return True


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    effect_type: EffectType
    realizability: Realizability
    enable_teacher_component: ComponentFn = _noop
    disable_student_component: ComponentFn = _noop
    event_detector: ComponentFn = _true
    build_teacher_view: ComponentFn = lambda state: state
    build_student_view: ComponentFn = lambda state: state
    snapshot_state: ComponentFn = lambda state: state
    restore_state: ComponentFn = lambda snapshot: snapshot
    effect_extractor: ComponentFn | None = None
    projection_builder: ComponentFn | None = None
    supervision_builder: ComponentFn = lambda record: record
    default_loss_mode: LossMode = "none"
    visibility_validator: ComponentFn = _true
    action_schema_validator: ComponentFn = _true
    leakage_validator: ComponentFn = _true
    mechanism_metrics: list[str] = field(default_factory=list)
    train_refusal_code: str | None = None

    def can_train(self, *, student_has_tool: bool = False) -> tuple[bool, str]:
        if self.realizability == "NON_REALIZABLE" and not student_has_tool:
            return False, self.train_refusal_code or "NON_REALIZABLE_ACTION_SPACE_MISMATCH"
        return True, "TRAINABLE"
