"""Shadow module base class — read-only analysis on DecisionState."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.artifacts.schema import PrivilegedArtifact
from harness.artifacts.validators import ValidationResult
from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState


class ShadowModule(ABC):
    """Typed shadow module. Must not mutate WorkingMemory or advance the env."""

    module_id: str
    schema_version: str = "shadow.v1"

    @abstractmethod
    def analyze(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> PrivilegedArtifact:
        ...

    @abstractmethod
    def validate_candidate(
        self,
        state: DecisionState,
        candidate: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        ...
