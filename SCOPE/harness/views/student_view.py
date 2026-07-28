"""Student view: deployment-visible context only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StudentView:
    query: str
    recent_trajectory: str
    module_state_text: str = ""

    def render(self) -> str:
        parts = [f"Query: {self.query}"]
        if self.module_state_text:
            parts.append(self.module_state_text)
        if self.recent_trajectory:
            parts.append(self.recent_trajectory)
        return "\n\n".join(parts)

    @classmethod
    def from_episode_state(
        cls,
        query: str,
        working_memory: Any,
        recent_trajectory: str,
        *,
        include_verification: bool = False,
    ) -> StudentView:
        module_state = ""
        if working_memory is not None:
            if hasattr(working_memory, "get_structured_state"):
                module_state = working_memory.get_structured_state(
                    include_verification=include_verification
                )
        return cls(
            query=query,
            recent_trajectory=recent_trajectory,
            module_state_text=module_state,
        )
