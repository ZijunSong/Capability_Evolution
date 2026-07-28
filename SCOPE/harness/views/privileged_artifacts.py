"""Privileged artifacts for OPD teacher views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrivilegedArtifacts:
    module_id: str
    turn_id: int
    compact_text: str
    structured_payload: dict[str, Any] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    future_leakage: bool = False

    def validate(self) -> None:
        if self.future_leakage:
            raise ValueError(
                f"Refusing privileged artifact with future_leakage=True "
                f"(module={self.module_id}, turn={self.turn_id})"
            )
