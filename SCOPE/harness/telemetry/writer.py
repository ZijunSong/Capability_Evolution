"""SCOPE event writer wrapping JsonlWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.telemetry.events import SCOPE_EVENT_TYPES, ScopeEvent, ScopeStats
from harness.telemetry.jsonl_writer import JsonlWriter


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
