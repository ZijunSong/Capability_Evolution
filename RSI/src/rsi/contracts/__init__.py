"""Type contracts shared across RSI (tasks, actions, observations, policies,
traces, artifacts).  Everything here is versioned and serializable."""

from __future__ import annotations

from rsi.contracts.action import (
    ACTION_SCHEMA_ID,
    Action,
    ActionParseError,
    Finish,
    ToolCall,
    encode_action,
    parse_action,
)
from rsi.contracts.artifact import (
    BankManifest,
    CandidateFile,
    CandidatePatch,
    CandidateRecord,
    ComponentManifest,
    empty_bank_manifest,
)
from rsi.contracts.common import (
    JSONValue,
    canonical_json,
    digest_of,
    digest_of_file,
    digest_of_text,
    short,
)
from rsi.contracts.errors import (
    BudgetExceeded,
    CandidateInvalid,
    ComponentRuntimeError,
    EnvironmentError_,
    InfraError,
    ModelInvalidAction,
    NotImplementedCapability,
    ProtocolViolation,
    RSIError,
)
from rsi.contracts.events import CostRecord, Event, EventKind, Visibility, total_cost
from rsi.contracts.observation import (
    ArtifactRef,
    Observation,
    ObservationProvenance,
    ScoreRecord,
    SnapshotRef,
    StepResult,
    ToolSpec,
    tool_catalog_digest,
)
from rsi.contracts.policy import (
    PolicyArtifact,
    RunIdentity,
    RuntimeContract,
    SamplingParams,
)
from rsi.contracts.task import (
    DATA_ROLES,
    PrivateTaskRef,
    PublicTask,
    SplitAssignment,
    SplitManifest,
)
from rsi.contracts.trace import (
    ALIGNMENT_ROUTES,
    AlignmentResult,
    DecisionRecord,
    EpisodeTrace,
    ProvenanceReport,
    RealizabilityReport,
)

# Alias kept for readability at call sites.
CandidateManifest = ComponentManifest

__all__ = [
    "ACTION_SCHEMA_ID",
    "Action",
    "ActionParseError",
    "Finish",
    "ToolCall",
    "encode_action",
    "parse_action",
    "BankManifest",
    "CandidateFile",
    "CandidateManifest",
    "CandidatePatch",
    "CandidateRecord",
    "ComponentManifest",
    "empty_bank_manifest",
    "JSONValue",
    "canonical_json",
    "digest_of",
    "digest_of_file",
    "digest_of_text",
    "short",
    "BudgetExceeded",
    "CandidateInvalid",
    "ComponentRuntimeError",
    "EnvironmentError_",
    "InfraError",
    "ModelInvalidAction",
    "NotImplementedCapability",
    "ProtocolViolation",
    "RSIError",
    "CostRecord",
    "Event",
    "EventKind",
    "Visibility",
    "total_cost",
    "ArtifactRef",
    "Observation",
    "ObservationProvenance",
    "ScoreRecord",
    "SnapshotRef",
    "StepResult",
    "ToolSpec",
    "tool_catalog_digest",
    "PolicyArtifact",
    "RunIdentity",
    "RuntimeContract",
    "SamplingParams",
    "DATA_ROLES",
    "PrivateTaskRef",
    "PublicTask",
    "SplitAssignment",
    "SplitManifest",
    "ALIGNMENT_ROUTES",
    "AlignmentResult",
    "DecisionRecord",
    "EpisodeTrace",
    "ProvenanceReport",
    "RealizabilityReport",
]
