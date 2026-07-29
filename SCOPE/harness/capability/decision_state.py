"""DecisionState V2 public entry point (re-exports from state.py).

The canonical implementation lives in ``harness.capability.state`` for backward
compatibility with existing imports.  New code may import from either module.
"""

from harness.capability.state import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V1,
    ActionRecord,
    ClaimState,
    DecisionState,
    DecisionStateV2,
    ObservationRecord,
    SourceType,
    VerificationRecordState,
    compute_text_hash,
    compute_wm_snapshot_hash,
)

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V1",
    "ActionRecord",
    "ClaimState",
    "DecisionState",
    "DecisionStateV2",
    "ObservationRecord",
    "SourceType",
    "VerificationRecordState",
    "compute_text_hash",
    "compute_wm_snapshot_hash",
]
