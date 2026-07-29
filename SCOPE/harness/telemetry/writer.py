"""SCOPE event writer wrapping JsonlWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from harness.telemetry.events import SCOPE_EVENT_TYPES, ScopeEvent, ScopeStats
from harness.telemetry.jsonl_writer import JsonlWriter

if TYPE_CHECKING:
    from harness.artifacts.gates import InformationSafeReport
    from harness.artifacts.schema import PrivilegedArtifact
    from harness.capability.action_space import CapabilityAction
    from harness.capability.state import DecisionState
    from training.scope.routing import RoutingResult


class ScopeTelemetryWriter:
    def __init__(self, path: str | Path) -> None:
        self._writer = JsonlWriter(Path(path))
        self.stats = ScopeStats()

    def emit(self, event: ScopeEvent) -> None:
        if event.event not in SCOPE_EVENT_TYPES:
            event.payload = {**event.payload, "unknown_event": event.event}
        self._writer.write(event.to_dict())

    def emit_dict(self, event_type: str, **kwargs: Any) -> None:
        episode_id = str(kwargs.pop("episode_id", ""))
        turn_id = int(kwargs.pop("turn_id", 0))
        module_id = kwargs.pop("module_id", None)
        self.emit(
            ScopeEvent(
                event=event_type,
                episode_id=episode_id,
                turn_id=turn_id,
                module_id=module_id,
                payload=kwargs,
            )
        )

    def flush_stats(self) -> dict[str, Any]:
        return self.stats.to_dict()

    def record_supervision_pipeline(
        self,
        *,
        state: "DecisionState",
        student_action: "CapabilityAction",
        artifact: "PrivilegedArtifact",
        routing: "RoutingResult",
        pre_gates: "InformationSafeReport | None" = None,
        event_id: str = "",
    ) -> None:
        """Emit a v3 supervision event with fields required by 0728-todo1 §15."""
        gates = routing.gates
        cap = artifact.resolved_capability().value
        route = routing.route.value
        sample = routing.sample

        self.stats.record_guidance(
            artifact.mode.value, artifact.module_id, artifact.reason_code
        )
        self.stats.record_capability_route(
            cap,
            route,
            visibility_violation=not gates.visible,
            shadow_mutation=not gates.purity_ok,
            invalid_action=not gates.executable,
            verifier_reject=bool(
                routing.target_validation is not None
                and not routing.target_validation.valid
            ),
        )
        if not gates.visible:
            self.stats.visibility_violations += 1
        if routing.candidate is not None:
            self.stats.candidate_total += 1
            if routing.target_validation and routing.target_validation.valid:
                self.stats.candidate_pass += 1

        payload: dict[str, Any] = {
            "event_id": event_id or state.event_id or f"{state.episode_id}:{state.turn_id}",
            "decision_state_hash": state.core_state_hash(),
            "capability_id": cap,
            "student_action": student_action.to_dict(),
            "artifact": artifact.to_dict(),
            "gate_results": gates.to_dict(),
            "pre_gate_results": pre_gates.to_dict() if pre_gates else None,
            "candidate_action": (
                routing.candidate.action.to_dict() if routing.candidate else None
            ),
            "target_action": (
                routing.target_action.to_dict() if routing.target_action else None
            ),
            "verifier_result": {
                "student": (
                    routing.student_validation.to_dict()
                    if routing.student_validation
                    else None
                ),
                "target": (
                    routing.target_validation.to_dict()
                    if routing.target_validation
                    else None
                ),
            },
            "route": route,
            "train_mask": sample.train_mask,
            "sample_id": sample.sample_id,
            "audit_error": sample.audit_error,
        }
        self.emit_dict(
            "supervision_sample_emitted",
            episode_id=state.episode_id,
            turn_id=state.turn_id,
            module_id=artifact.module_id,
            **payload,
        )
