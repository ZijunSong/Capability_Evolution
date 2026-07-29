"""SCOPE v3 validators: information-safe gates + local action verifiers.

This module is the training-side façade over:
  - ``harness.artifacts.gates`` (InformationSafeGate)
  - ``harness.artifacts.validators`` (module-local verifiers)
"""

from __future__ import annotations

from harness.artifacts.gates import (
    InformationSafeReport,
    capture_env_fingerprint,
    executability_gate,
    module_responsibility_gate,
    run_information_safe_gates,
    runtime_provenance_gate,
    schema_gate,
    shadow_purity_gate,
    visibility_gate,
)
from harness.artifacts.schema import PrivilegedArtifact
from harness.artifacts.validators import (
    BudgetVerifier,
    EvidenceVerifier,
    LocalVerifier,
    ValidationResult,
    VerificationVerifier,
    get_verifier,
)
from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState

# Semantic alias used in 0728-todo1
InformationSafeGate = run_information_safe_gates

__all__ = [
    "BudgetVerifier",
    "EvidenceVerifier",
    "InformationSafeGate",
    "InformationSafeReport",
    "LocalVerifier",
    "ValidationResult",
    "VerificationVerifier",
    "capture_env_fingerprint",
    "executability_gate",
    "get_verifier",
    "module_responsibility_gate",
    "run_information_safe_gates",
    "runtime_provenance_gate",
    "schema_gate",
    "shadow_purity_gate",
    "validate_recommended_action",
    "visibility_gate",
]


def validate_recommended_action(
    state: DecisionState,
    artifact: PrivilegedArtifact,
    action: CapabilityAction | None = None,
) -> ValidationResult:
    """Run the module-local verifier on a candidate / recommended action."""
    candidate = action or artifact.recommended_action
    if candidate is None:
        return ValidationResult(valid=False, score=0.0, reasons=("no_candidate_action",))
    verifier = get_verifier(artifact.module_id)
    return verifier.validate(state, candidate, artifact)
