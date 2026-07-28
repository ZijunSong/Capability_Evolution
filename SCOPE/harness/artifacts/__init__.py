"""SCOPE privileged artifact schemas and guards."""

from harness.artifacts.provenance import ProvenanceKind, assert_info_subset
from harness.artifacts.reason_codes import (
    BUDGET_REASON_CODES,
    EVIDENCE_REASON_CODES,
    VERIFICATION_REASON_CODES,
)
from harness.artifacts.schema import GuidanceMode, LocalDecisionArtifact, PrivilegedArtifact
from harness.artifacts.validators import LocalVerifier, ValidationResult
from harness.artifacts.visibility import VisibilityCheck, check_artifact_visibility

# gates imported lazily via harness.artifacts.gates to avoid circular imports
# with harness.capability.state → provenance → artifacts.__init__ → gates → state

__all__ = [
    "BUDGET_REASON_CODES",
    "EVIDENCE_REASON_CODES",
    "VERIFICATION_REASON_CODES",
    "GuidanceMode",
    "LocalDecisionArtifact",
    "LocalVerifier",
    "PrivilegedArtifact",
    "ProvenanceKind",
    "ValidationResult",
    "VisibilityCheck",
    "assert_info_subset",
    "check_artifact_visibility",
]


def __getattr__(name: str):
    if name in {
        "InformationSafeReport",
        "run_information_safe_gates",
        "capture_env_fingerprint",
    }:
        from harness.artifacts import gates as _gates

        return getattr(_gates, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
